from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


class GOLDLossAdapter(nn.Module):
    """
    - GOLD (General Online Logit Distillation) 损失函数适配器
    支持:
    1. ULD损失 (Universal Logit Distillation)
    2. 扩展ULD (跨tokenizer对齐)
    3. 混合损失 (Hybrid ULD + JSD)

    使用示例:
        adapter = GOLDLossAdapter(
            config={
                'use_uld_loss': True,
                'use_extended_uld': True,
                'uld_use_hybrid_loss': False,
                'uld_crossentropy_weight': 0.0,
                'uld_distillation_weight': 1.0,
                'uld_student_temperature': 1.0,
                'uld_teacher_temperature': 1.0,
            },
            student_tokenizer=student_tok,
            teacher_tokenizer=teacher_tok,
        )

        loss = adapter(
            student_logits=student_outputs.logits,
            teacher_logits=teacher_outputs.logits,
            student_labels=student_labels,
            teacher_labels=teacher_labels,
            student_input_ids=student_input_ids,
            teacher_input_ids=teacher_input_ids,
        )
    """

    def __init__(
            self,
            config: dict,
            student_tokenizer: Optional[PreTrainedTokenizerBase] = None,
            teacher_tokenizer: Optional[PreTrainedTokenizerBase] = None,
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device

        # 基础配置
        self.use_uld_loss = config.get('use_uld_loss', True)  # 是否开启通用蒸馏
        self.crossentropy_weight = config.get('uld_crossentropy_weight', 0.0)
        self.distillation_weight = config.get('uld_distillation_weight', 1.0)
        self.student_temperature = config.get('uld_student_temperature', 0.9)
        self.teacher_temperature = config.get('uld_teacher_temperature', 0.9)
        self.skip_student_eos = config.get('uld_skip_student_eos', True)
        self.skip_teacher_eos = config.get('uld_skip_teacher_eos', True)
        self.use_extended_uld = config.get('use_extended_uld', True)
        self.ignore_index = -100

        # Tokenizers
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        # Hybrid ULD配置
        self.use_hybrid_loss = config.get('uld_use_hybrid_loss', True)  # 是否对完全匹配的词汇进行匹配，开启提高稳定性
        self.hybrid_matched_weight = config.get('uld_hybrid_matched_weight', None)
        self.hybrid_unmatched_weight = config.get('uld_hybrid_unmatched_weight', None)
        self.beta = config.get('beta', 1.0)

        # 初始化词汇映射（用于hybrid loss）
        self._vocab_mapping = None
        self._teacher_matched_ids = None
        self._student_matched_ids = None
        self.mapping_tensor = None

        if self.use_hybrid_loss and student_tokenizer and teacher_tokenizer:
            self._initialize_vocabulary_mapping()

        # 用于logging
        self.last_matched_loss = None
        self.last_unmatched_loss = None

    def _initialize_vocabulary_mapping(self):
        """初始化学生-教师tokenizer的词汇映射"""
        student_vocab = self.student_tokenizer.get_vocab()
        teacher_vocab = self.teacher_tokenizer.get_vocab()

        student_token_to_id = dict(student_vocab.items())

        vocab_mapping = {}
        teacher_matched_ids = set()
        student_matched_ids = set()

        for token_str, teacher_id in teacher_vocab.items():
            if token_str in student_token_to_id:
                student_id = student_token_to_id[token_str]
                vocab_mapping[teacher_id] = student_id
                teacher_matched_ids.add(teacher_id)
                student_matched_ids.add(student_id)

        self._vocab_mapping = vocab_mapping
        self._teacher_matched_ids = teacher_matched_ids
        self._student_matched_ids = student_matched_ids

        if self._vocab_mapping:
            max_matched_teacher_id = max(self._vocab_mapping.keys())
            self.mapping_tensor = torch.full(
                (max_matched_teacher_id + 1,), -1, dtype=torch.long
            )
            for k, v in self._vocab_mapping.items():
                self.mapping_tensor[k] = v
            if self.device:
                self.mapping_tensor = self.mapping_tensor.to(self.device)

    def forward(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            student_labels: torch.Tensor,
            teacher_labels: torch.Tensor,
            student_input_ids: torch.Tensor,
            teacher_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算GOLD/ULD损失

        Args:
            student_logits: [batch_size, seq_len, student_vocab_size]
            teacher_logits: [batch_size, seq_len, teacher_vocab_size]
            student_labels: [batch_size, seq_len], -100表示忽略
            teacher_labels: [batch_size, seq_len], -100表示忽略
            student_input_ids: [batch_size, seq_len]
            teacher_input_ids: [batch_size, seq_len]

        Returns:
            loss: scalar tensor
        """

        if not self.use_uld_loss:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        # 1. Cross-entropy loss (可选，通过crossentropy_weight设置权重)
        crossentropy_loss = self._compute_cross_entropy(student_logits, student_labels)

        # 2. Distillation loss (ULD)
        distillation_loss = self._compute_distillation_loss(
            student_logits, teacher_logits,
            student_labels, teacher_labels,
            student_input_ids, teacher_input_ids
        )
        return crossentropy_loss + distillation_loss

    def _compute_cross_entropy(
            self,
            student_logits: torch.Tensor,
            student_labels: torch.Tensor
    ) -> torch.Tensor:
        """计算cross-entropy loss"""
        if self.crossentropy_weight <= 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = student_labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        ce_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return self.crossentropy_weight * ce_loss

    def _compute_distillation_loss(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            student_labels: torch.Tensor,
            teacher_labels: torch.Tensor,
            student_input_ids: torch.Tensor,
            teacher_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """计算ULD蒸馏损失"""
        # 获取答案区域
        student_answer_idx, student_answer_size = self._get_answer_regions(student_labels)
        teacher_answer_idx, teacher_answer_size = self._get_answer_regions(teacher_labels)

        if self.skip_student_eos:
            student_answer_size = [s - 1 for s in student_answer_size]
        if self.skip_teacher_eos:
            teacher_answer_size = [t - 1 for t in teacher_answer_size]

        # 边界检查
        if not student_answer_size or not teacher_answer_size:
            return torch.zeros(1, device=student_logits.device, requires_grad=True) * 1e-8

        batch_size = student_logits.size(0)
        distillation_losses = []

        for i in range(batch_size):
            s_start = student_answer_idx[i]
            s_size = student_answer_size[i]
            t_start = teacher_answer_idx[i]
            t_size = teacher_answer_size[i]

            if s_size <= 0 or t_size <= 0:
                loss_i = student_logits[i].sum() * 0.0
                # Ensure the loss tensor requires gradients
                loss_i = loss_i.detach().requires_grad_(True)
                distillation_losses.append(loss_i)
                continue

            # 提取答案logits
            student_ans_logits = student_logits[i, s_start:s_start + s_size]
            teacher_ans_logits = teacher_logits[i, t_start:t_start + t_size]

            # 转换为概率
            student_probs = F.softmax(student_ans_logits / self.student_temperature, dim=-1)
            teacher_probs = F.softmax(teacher_ans_logits / self.teacher_temperature, dim=-1)

            student_token_ids = student_input_ids[i, s_start:s_start + s_size].tolist()
            teacher_token_ids = teacher_input_ids[i, t_start:t_start + t_size].tolist()

            # Token对齐
            if self.use_extended_uld:
                student_groups, teacher_groups = self._build_alignment_groups_from_ids(
                    student_token_ids, teacher_token_ids
                )

                student_aligned = self._merge_probabilities_with_groups(
                    student_probs, student_groups, student_token_ids
                )
                teacher_aligned = self._merge_probabilities_with_groups(
                    teacher_probs, teacher_groups, teacher_token_ids
                )

            else:
                min_len = min(len(student_token_ids), len(teacher_token_ids))
                student_aligned = student_probs[:min_len]
                teacher_aligned = teacher_probs[:min_len]

            # 计算损失
            if self.use_hybrid_loss and self._vocab_mapping:
                aligned_loss = self._compute_hybrid_uld_loss(student_aligned, teacher_aligned)
            else:
                aligned_loss = self._compute_basic_uld_loss(student_aligned, teacher_aligned)

            distillation_losses.append(aligned_loss)
        distillation_loss = torch.stack(distillation_losses).mean()
        return self.distillation_weight * distillation_loss

    def _get_answer_regions(self, labels: torch.Tensor) -> Tuple[List[int], List[int]]:
        """获取答案区域的起始位置和大小"""
        indices = []
        sizes = []

        for label in labels:
            mask = label.ne(self.ignore_index)
            if not mask.any():
                indices.append(0)
                sizes.append(0)
                continue

            valid_indices = mask.nonzero(as_tuple=True)[0]
            indices.append(int(valid_indices[0].item()))
            sizes.append(int(mask.sum().item()))

        return indices, sizes

    def _build_alignment_groups_from_ids(
            self,
            student_token_ids: List[int],
            teacher_token_ids: List[int]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        基于文本内容构建对齐组
        使用贪心子串匹配算法
        """

        def decode_tokens(tokenizer, token_ids):
            pieces = []
            prev = ""
            for k in range(len(token_ids)):
                cur = tokenizer.decode(token_ids[:k + 1], skip_special_tokens=False)
                pieces.append(cur[len(prev):])
                prev = cur
            return pieces

        student_pieces = decode_tokens(self.student_tokenizer, student_token_ids)
        teacher_pieces = decode_tokens(self.teacher_tokenizer, teacher_token_ids)

        # 贪心匹配算法
        student_groups = []
        teacher_groups = []
        s_idx = 0
        t_idx = 0

        while s_idx < len(student_pieces) and t_idx < len(teacher_pieces):
            student_text = ""
            teacher_text = ""
            student_group = []
            teacher_group = []

            # 尝试找到最短的连续匹配序列
            while s_idx < len(student_pieces) and t_idx < len(teacher_pieces):
                if not student_group:
                    student_group.append(s_idx)
                    student_text += student_pieces[s_idx]
                    s_idx += 1

                if not teacher_group:
                    teacher_group.append(t_idx)
                    teacher_text += teacher_pieces[t_idx]
                    t_idx += 1

                # 检查是否匹配
                if student_text == teacher_text:
                    student_groups.append(student_group)
                    teacher_groups.append(teacher_group)
                    break
                elif len(student_text) < len(teacher_text):
                    if s_idx < len(student_pieces):
                        student_group.append(s_idx)
                        student_text += student_pieces[s_idx]
                        s_idx += 1
                    else:
                        break
                else:
                    if t_idx < len(teacher_pieces):
                        teacher_group.append(t_idx)
                        teacher_text += teacher_pieces[t_idx]
                        t_idx += 1
                    else:
                        break
            else:
                # 未完全匹配，添加剩余部分
                if student_group and teacher_group:
                    student_groups.append(student_group)
                    teacher_groups.append(teacher_group)

        return student_groups, teacher_groups

    def _merge_probabilities_with_groups(
            self,
            probs: torch.Tensor,
            alignment_groups: List[List[int]],
            token_ids: List[int],
    ) -> torch.Tensor:
        """
        根据对齐组合并概率分布
        使用链式法则: P_merged = P(y|x_0) * P(x_1|x_0) * P(x_2|x_0,x_1) * ...
        """
        aligned_probs = []

        for group in alignment_groups:
            if len(group) > 1:
                # 第一个token的边际概率
                marginal_probs = probs[group[0]]  # [vocab_size]

                # 后续token的条件概率（标量）
                conditional_product = 1.0
                for k in range(1, len(group)):
                    cond_prob = probs[group[k], token_ids[group[k - 1]]]
                    conditional_product *= cond_prob

                merged_probs = marginal_probs * conditional_product
                aligned_probs.append(merged_probs)
            elif len(group) == 1:
                aligned_probs.append(probs[group[0]])

        if aligned_probs:
            return torch.stack(aligned_probs)
        else:
            # 返回一个空的但需要梯度的张量
            empty_tensor = probs[:0].detach().requires_grad_(True)
            return empty_tensor

    def _compute_basic_uld_loss(
            self,
            student_aligned: torch.Tensor,
            teacher_aligned: torch.Tensor,
    ) -> torch.Tensor:
        """基础ULD损失：排序后的L1距离"""
        student_sorted = student_aligned.sort(dim=-1, descending=True).values
        teacher_sorted = teacher_aligned.sort(dim=-1, descending=True).values

        # Padding到相同vocab size
        s_vocab = student_sorted.size(-1)
        t_vocab = teacher_sorted.size(-1)
        max_vocab = max(s_vocab, t_vocab)

        if s_vocab < max_vocab:
            student_sorted = F.pad(student_sorted, (0, max_vocab - s_vocab))
        if t_vocab < max_vocab:
            teacher_sorted = F.pad(teacher_sorted, (0, max_vocab - t_vocab))

        loss = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
        loss /= student_aligned.size(0)

        return loss

    def _compute_hybrid_uld_loss(
            self,
            student_aligned: torch.Tensor,
            teacher_aligned: torch.Tensor,
    ) -> torch.Tensor:
        """混合ULD损失：matched用JSD，unmatched用排序L1"""
        device = student_aligned.device
        s_vocab = student_aligned.size(-1)
        t_vocab = teacher_aligned.size(-1)

        # 创建matched/unmatched masks
        if self._teacher_matched_ids:
            teacher_matched_idx = torch.tensor(
                sorted(self._teacher_matched_ids), dtype=torch.long, device=device
            )
            student_matched_idx = self.mapping_tensor[teacher_matched_idx]
        else:
            teacher_matched_idx = torch.tensor([], dtype=torch.long, device=device)
            student_matched_idx = torch.tensor([], dtype=torch.long, device=device)

        teacher_matched_mask = torch.zeros(t_vocab, dtype=torch.bool, device=device)
        student_matched_mask = torch.zeros(s_vocab, dtype=torch.bool, device=device)

        if len(teacher_matched_idx) > 0:
            teacher_matched_mask[teacher_matched_idx] = True
            student_matched_mask[student_matched_idx] = True

        # 1. Matched tokens的JSD损失
        matched_loss = torch.tensor(0.0, device=device, requires_grad=True)
        matched_count = 0

        if len(teacher_matched_idx) > 0:
            teacher_matched_probs = teacher_aligned[:, teacher_matched_idx]
            student_matched_probs = student_aligned[:, student_matched_idx]
            matched_count = teacher_matched_probs.size(-1)

            matched_loss = self._compute_jsd_for_matched(
                student_matched_probs, teacher_matched_probs
            )
        # 2. Unmatched tokens的排序L1损失
        teacher_unmatched = teacher_aligned[:, ~teacher_matched_mask]
        student_unmatched = student_aligned[:, ~student_matched_mask]

        unmatched_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if teacher_unmatched.size(-1) > 0 and student_unmatched.size(-1) > 0:
            teacher_sorted = teacher_unmatched.sort(dim=-1, descending=True).values
            student_sorted = student_unmatched.sort(dim=-1, descending=True).values

            t_size = teacher_sorted.size(-1)
            s_size = student_sorted.size(-1)
            max_size = max(t_size, s_size)

            if t_size < max_size:
                teacher_sorted = F.pad(teacher_sorted, (0, max_size - t_size))
            if s_size < max_size:
                student_sorted = F.pad(student_sorted, (0, max_size - s_size))

            unmatched_loss = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
            unmatched_loss /= student_aligned.size(0)

        # 3. 加权组合
        if self.hybrid_matched_weight is None:
            w_matched = matched_count / max(1, t_vocab)
            w_unmatched = 1.0 - w_matched
        else:
            w_matched = self.hybrid_matched_weight
            w_unmatched = self.hybrid_unmatched_weight

        total_loss = w_matched * matched_loss + w_unmatched * unmatched_loss

        # 保存用于logging
        self.last_matched_loss = matched_loss
        self.last_unmatched_loss = unmatched_loss

        return total_loss

    def _compute_jsd_for_matched(
            self,
            student_probs: torch.Tensor,
            teacher_probs: torch.Tensor,
            epsilon: float = 1e-8
    ) -> torch.Tensor:
        """计算matched tokens的JSD损失，添加数值稳定性处理"""
        batch_seq_len, num_matched = student_probs.shape

        # 检查输入概率分布是否有效
        if torch.isnan(student_probs).any() or torch.isnan(teacher_probs).any():
            return torch.tensor(0.0, device=student_probs.device, requires_grad=True)

        # 添加epsilon防止数值下溢和log(0)
        student_probs = student_probs.clamp(min=epsilon)
        teacher_probs = teacher_probs.clamp(min=epsilon)

        # 重新归一化概率分布
        student_probs = student_probs / student_probs.sum(dim=-1, keepdim=True)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)

        student_flat = student_probs.view(-1, num_matched)
        teacher_flat = teacher_probs.view(-1, num_matched)

        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
        m = 0.5 * (student_flat + teacher_flat)

        # 添加epsilon到中间分布
        m = m.clamp(min=epsilon)
        m = m / m.sum(dim=-1, keepdim=True)

        # 直接对概率分布取对数，添加epsilon防止数值问题
        log_m = torch.log(m + epsilon)
        log_student = torch.log(student_flat + epsilon)
        log_teacher = torch.log(teacher_flat + epsilon)

        # 使用log_target=True，传入log概率
        kl_p_m = F.kl_div(log_m, log_student, reduction='batchmean', log_target=True)
        kl_q_m = F.kl_div(log_m, log_teacher, reduction='batchmean', log_target=True)
        jsd = 0.5 * (kl_p_m + kl_q_m)

        # 检查结果是否有效
        if torch.isnan(jsd) or torch.isinf(jsd):
            return torch.tensor(0.0, device=student_probs.device, requires_grad=True)

        return jsd
