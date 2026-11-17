# Copyright (c) ModelScope Contributors. All rights reserved.
"""v5 原生载荷契约：公共样本载荷只有 Trajectory，公共前向结果只有 ModelOutput。

字段声明不创建 tensor 包装或复制重载荷；profile 是同一原生载荷的字段能力阶段，
不是新包装。前向角色（ref / teacher / old / value）由消费口表示，MODEL_OUTPUT
只携带字段能力。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ChannelField:
    name: str
    doc: str
    dtype: str = ''
    frame: str = ''
    required: bool = True
    guarded: bool = False


@dataclass(frozen=True)
class ChannelProfile:
    profile: str
    doc: str
    fields: tuple[ChannelField, ...] = field(default=())
    cardinality: str = ''
    alignment: str = ''
    layout: str = ''
    references: str = ''


@dataclass(frozen=True)
class ChannelSchema:
    kind: str
    doc: str
    reference: bool = False
    profiles: tuple[ChannelProfile, ...] = field(default=())

    def profile(self, name: str) -> ChannelProfile:
        for profile in self.profiles:
            if profile.profile == name:
                return profile
        raise KeyError(f'{self.kind}: 未声明 profile {name!r}')

    def to_dict(self) -> dict:
        value = asdict(self)
        value['profiles'] = [
            {**asdict(profile), 'fields': [asdict(column) for column in profile.fields]}
            for profile in self.profiles]
        value['allows'] = [self.kind]
        return value


# 身份在控制元数据中保存；同形状不意味着相同行或相同 token。
IDENTITY_FIELDS = (
    ChannelField('batch_id', 'source 分配的稳定批次身份', 'str', guarded=True),
    ChannelField('row_ids', '按实际行顺序排列的稳定身份', 'str[B]', 'row', guarded=True),
    ChannelField('group_ids', '显式完整分组身份', 'str[B]', 'row', required=False, guarded=True),
    ChannelField('pair_ids', '成对数据的原子身份', 'str[B]', 'row', required=False, guarded=True),
)
TOKEN_FIELDS = (
    ChannelField('tokenizer_fingerprint', 'tokenizer 与编码规则指纹', 'str', guarded=True),
    ChannelField('response_token_ids', '每行实际响应 token 身份', 'int64[T_i]', 'response', guarded=True),
    ChannelField('response_loss_mask', '与响应 token 一一对应的训练 mask', 'bool[T_i]', 'response', guarded=True),
    ChannelField('token_frame', '轮次、输入与响应 token 的位置映射', 'dict[B]', guarded=True),
)
ALIGNMENT = 'batch_id + row_ids/order + group_ids/pair_ids（若有）'
TOKEN_ALIGNMENT = ALIGNMENT + ' + tokenizer_fingerprint + response_token_ids + response_loss_mask + token_frame'
REFERENCES = 'local 原生值或 infra 字段/分片引用；计算 worker resolve，driver 仅持小型身份与完成回执'
# 前向输出的来源事实：哪个模型、哪个参数版本。
OUTPUT_SOURCE = (
    ChannelField('model_id', '前向模型对象身份；纯数值结果保留来源集合', 'str|str[]', required=False),
    ChannelField('parameter_version', '实际前向所用模型版本', 'int|int[]', required=False),
)
FEATURE_FIELDS = (
    ChannelField('input_ids', '完整序列 token', 'int64[L_i]', 'input'),
    ChannelField('labels', '原生编码规则的训练标签', 'int64[L_i]', 'input', required=False),
    ChannelField('attention_mask', '原生 attention mask', 'bool[L_i]', 'input', required=False),
    ChannelField('completion_mask', '响应位置 mask', 'bool[L_i]', 'input', required=False),
    ChannelField('loss_scale', 'token 损失权重', 'float32[L_i]', 'input', required=False),
    ChannelField('extensions', '多模态/MoE 等原生 feature 扩展列', 'dict', required=False),
)
#: TRAJ 的能力阶段字段组：同一原生轨迹，字段逐步就位。
RAW_FIELDS = (
    ChannelField('messages', '保留完整多轮消息及工具结果', 'messages[B]', 'row'),
    ChannelField('extra', '原生扩展字段，不丢弃多模态和业务列', 'dict', required=False),
)
ROLLOUT_FIELDS = (
    ChannelField('encoded', '原生完整输入 feature；labels shift 由编码规则决定', 'InputFeature[L_i]', 'input', guarded=True),
    ChannelField('rollout_logps', '实际采样 logps；引擎不提供时明确缺失', 'float32[T_i]', 'response', required=False, guarded=True),
    ChannelField('messages', '保留完整多轮消息及工具结果', 'messages[B]', 'row'),
    ChannelField('sampling_masks', '采样支持集合', 'SamplingMask[T_i]', 'response', required=False),
    ChannelField('policy_version', '控制面准入时确定的参数版本', 'int', required=False),
    ChannelField('extra', '原生扩展字段，不丢弃多模态和业务列', 'dict', required=False),
)
SCORED_FIELDS = ROLLOUT_FIELDS + (
    ChannelField('rewards', '具名评分列：orm 具名 scorer→float|null、env float、baseline 具名 scorer→float|null',
                 'dict', required=False, guarded=True),
    ChannelField('prm_segments', 'PRM 段原始事实：段结束轮与具名原始分数', 'dict', required=False, guarded=True),
)
TRAIN_FIELDS = SCORED_FIELDS + (
    ChannelField('advantages', '响应帧优势', 'float32[T_i]', 'response', required=False),
    ChannelField('returns', '响应帧回报', 'float32[T_i]', 'response', required=False),
    ChannelField('old_logps', '采样或更新前策略 logps', 'float32[T_i]', 'response', required=False),
    ChannelField('ref_logps', '参考前向 logps', 'float32[T_i]', 'response', required=False),
    ChannelField('teacher_logps', '教师前向 logps', 'float32[T_i]', 'response', required=False),
    ChannelField('old_values', '更新前 critic 估值快照', 'float32[T_i]', 'response', required=False),
)


def _profile(name: str, doc: str, columns: tuple[ChannelField, ...], *,
             frame: str = 'row', layout: str = '', token: bool = False) -> ChannelProfile:
    return ChannelProfile(name, doc, (*columns, *IDENTITY_FIELDS, *(TOKEN_FIELDS if token else ())),
                          'B 行；ragged token 长度 L_i/T_i' if frame != 'global' else '1',
                          TOKEN_ALIGNMENT if token else ALIGNMENT, layout, REFERENCES)


def _scalar_profile(name: str, doc: str, columns: tuple[ChannelField, ...],
                    layout: str = '') -> ChannelProfile:
    return ChannelProfile(name, doc, columns, '1', '显式全局 frame', layout, REFERENCES)


CHANNEL_SCHEMAS: dict[str, ChannelSchema] = {
    'MODEL': ChannelSchema('MODEL', '唯一原生模型对象；冻结和放置是构造配置，disable_lora 是调用配置。', True),
    'DATASET': ChannelSchema('DATASET', '原生数据集或惰性数据来源，不属于循环 frame。', True),
    'SAMPLER': ChannelSchema('SAMPLER', '常驻采样引擎及实际已确认参数版本。', True),
    'ROLLOUT': ChannelSchema('ROLLOUT', '使用 sampler/template/tools/env 的 episode 生成机制。', True),
    'GROUP_RULE': ChannelSchema('GROUP_RULE', '完整分组的准入、筛选与有界重采规则，不做基线前向。', True),
    'ENV': ChannelSchema('ENV', '原生环境池/lease，工具实现按每次 lease 构造。', True),
    'TOOL': ChannelSchema('TOOL', '实际工具声明及插件来源；声明不等同于运行实现。', True),
    'ENDPOINT': ChannelSchema('ENDPOINT', '就绪服务或明确外部 API 连接；凭据只存在运行环境。', True),
    'SOURCE': ChannelSchema('SOURCE', '常驻 source 控制句柄；消费该口不触发 take。', True),
    'LOSS': ChannelSchema('LOSS', '损失对象：注册名构造的原生 Loss 或用户公式包装；train_step 安装后在前向内计算。', True),
    'SCORER': ChannelSchema('SCORER', '评分器对象：规则注册名构造或用户代码；分数以具名列写回轨迹。', True),
    'DATA': ChannelSchema('DATA', '未编码的原生行集合。', profiles=(
        _profile('raw', '原始或预测行；业务列原样保留。', RAW_FIELDS, layout='list[dict] 或 Dataset slice'),
        _profile('batch', 'Iterator 分配身份的当前批次。', RAW_FIELDS, layout='list[dict]'),
    )),
    'BATCH': ChannelSchema('BATCH', '原生编码 feature，身份元数据与重载荷分离。', profiles=(
        _profile('feature', '单行编码输入。', FEATURE_FIELDS, token=True, layout='list[InputFeature]'),
        _profile('paired', '完整 chosen/rejected 偏好对，选择不能拆对。',
                 (*FEATURE_FIELDS, ChannelField('pair_side', 'chosen/rejected 的原生排列', 'str[B]', 'row')),
                 token=True, layout='原生 paired feature 集；2P 行，pair_ids 标记原子对'),
        _profile('unpaired', '带显式偏好标签的编码行。',
                 (*FEATURE_FIELDS, ChannelField('label', '偏好标签', 'bool[B]', 'row')),
                 token=True, layout='list[InputFeature]，标签保持原生字段'),
    )),
    'TRAJ': ChannelSchema('TRAJ', '唯一公共轨迹载荷；profile 是同一原生轨迹的字段能力阶段。', profiles=(
        _profile('raw', '未编码的原始/预测行。', RAW_FIELDS, layout='list[dict]'),
        _profile('encoded', '已按编码规则落成完整输入 feature 的行。', FEATURE_FIELDS, token=True, layout='list[InputFeature]'),
        _profile('rollout', '生成轨迹：响应 token 身份与采样 logps 已就位。', ROLLOUT_FIELDS, token=True, layout='list[Trajectory]'),
        _profile('scored', 'rollout + 具名评分列（rewards.orm/baseline/env）。', SCORED_FIELDS, token=True, layout='list[Trajectory]'),
        _profile('train_ready', 'scored + 训练目标列（advantages/returns/old_logps/ref_logps/teacher_logps/old_values）。',
                 TRAIN_FIELDS, token=True, layout='list[Trajectory]'),
    )),
    'MODEL_OUTPUT': ChannelSchema('MODEL_OUTPUT', '前向结果或纯数值输出；字段能力是 profile，前向角色由消费口表示。', profiles=(
        _profile('logps', '逐 token log-probabilities（响应或完整输入 frame，按请求声明）。',
                 (ChannelField('values', '逐 token logps', 'float32[B,T_i]|float32[B,L_i]', 'response|input'),
                  *OUTPUT_SOURCE), token=True),
        _profile('values', 'value head 逐 token 估值（响应 frame）。',
                 (ChannelField('values', '逐 token 估值', 'float32[B,T_i]', 'response'), *OUTPUT_SOURCE), token=True),
        _profile('logits', '完整词表输出（输入 frame），不允许 top-k 冒充。',
                 (ChannelField('values', '全词表 logits', 'float32[B,L,V]', 'input'), *OUTPUT_SOURCE), token=True),
        _profile('embeddings', '原生隐藏表示。',
                 (ChannelField('values', '隐藏表示', 'float32[B,L,H]|float32[B,H]', 'input|row'), *OUTPUT_SOURCE), token=True),
        _profile('row', '每行数值归约结果。',
                 (ChannelField('values', '逐行数值', 'float32[B]', 'row'), *OUTPUT_SOURCE)),
        _profile('token', '明确响应 frame 的数值变换结果。',
                 (ChannelField('values', '逐 token 数值', 'float32[B,T_i]', 'response'), *OUTPUT_SOURCE), token=True),
        _scalar_profile('scalar', '明确全局 frame 的标量。',
                        (ChannelField('values', '有限标量', 'float32', 'global'),), layout='worker scalar 或轻量标量'),
    )),
    'CTRL': ChannelSchema('CTRL', '成功调用完成后的正向顺序 token；只有显式 frame feedback 是回边。'),
    'STEP': ChannelSchema('STEP', '真实训练完成回执，不把提交视作更新。', profiles=(
        _scalar_profile('step', '模型共享步数与本次批次消费状态。', (
            ChannelField('model_id', '实际更新模型身份', 'str'),
            ChannelField('micro_step', '已确认 backward 微步数', 'int'),
            ChannelField('global_step', '该模型已确认 optimizer 总步数', 'int'),
            ChannelField('optimizer_steps', '本次实际完成更新数', 'int'),
            ChannelField('batch_id', '所属当前批次', 'str'),
            ChannelField('consumed', '本消费者是否已完整消费', 'bool'),
        )),
    )),
    'SYNC': ChannelSchema('SYNC', '已完成并确认的参数传输；pending intent 只存在控制面。', profiles=(
        _scalar_profile('outcome', '真实版本与接收者确认。', (
            ChannelField('parameter_version', '实际同步参数版本', 'int'),
            ChannelField('receivers', '实际接收资源身份', 'str[]'),
            ChannelField('confirmed', '各接收者完成结果', 'bool[]'),
        )),
    )),
    'ARTIFACT': ChannelSchema('ARTIFACT', '实际写出的产物或经明确授权发布的远端产物。', profiles=tuple(
        _scalar_profile(name, doc, (
            ChannelField('path', '真实路径或发布 URI', 'str'),
            ChannelField('format', '实际格式', 'str'),
            ChannelField('manifest', '真实 manifest 路径（若有）', 'str', required=False)))
        for name, doc in (('model', '模型权重，不承诺完整续训。'), ('training', '安全边界保存的完整训练快照。'),
                          ('dataset', '数据或预测产物。'), ('report', '评测报告。'), ('publication', '实际 Hub 发布结果。')))),
    'METRIC': ChannelSchema('METRIC', '小型数值摘要；无效值为 null 并单独报告 invalid_count。', profiles=(
        _scalar_profile('summary', '具名标量。', (
            ChannelField('values', '标量映射', 'dict[str,float|null]'),
            ChannelField('invalid_count', '非有限值数量', 'int'))),
    )),
}

# kind 级提示只表达同 kind；跨 kind 适配必须读取消费者 PortSpec.accepted_kinds。
ALLOWS = {kind: (kind,) for kind in CHANNEL_SCHEMAS}


def allows(src_kind: str, dst_kind: str) -> bool:
    return src_kind in CHANNEL_SCHEMAS and src_kind == dst_kind


def profile_fields(kind: str, profile: str) -> tuple[ChannelField, ...]:
    return CHANNEL_SCHEMAS[kind].profile(profile).fields
