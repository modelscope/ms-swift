import os
import uuid
import re
import json
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
import glob
import os
from tqdm import tqdm

class CardValidator:
    """卡片内容验证器"""
    
    # 禁止的XML标签
    FORBIDDEN_TAGS = [
        '<strong>', '<b>', '<i>', '<em>', '<br>', '<p>', '<div>', '<img>', '<a>', 
        '<ul>', '<li>', '<ol>', '<table>', '<tr>', '<td>', '<th>', '<tbody>', 
        '<thead>', '<tfoot>', '<caption>', '<colgroup>', '<col>', '<style>', 
        '<script>', '<link>', '<meta>', '<base>', '<frame>', '<frameset>', 
        '<iframe>', '<noframes>', '<noscript>', '<noembed>', '<applet>', 
        '<embed>', '<object>', '<param>', '<map>', '<area>'
    ]
    
    def __init__(self):
        self.validation_result = {
            "valid": True,
            "errors": [],
            "extracted": {},
            'output_type': None
        }

    def _reset_validation(self) -> None:
        """重置验证结果"""
        self.validation_result = {
            "valid": True,
            "errors": [],
            "extracted": {}
        }
    
    def _add_error(self, msg: str) -> None:
        """添加错误信息"""
        self.validation_result["valid"] = False
        self.validation_result["errors"].append(msg)

    def _check_print_tag_content(self, ssw_card, context: str = "前置回复") -> None:
        """检查print标签内容中不应包含链接"""
        print_tag = ssw_card.find('print')
        if print_tag and print_tag.text:
            self._check_no_links_in_print(print_tag.text, context)

    def _find_and_validate_box(self, ssw_card, box_type: str, error_msg: str = None) -> Optional[Any]:
        """查找并验证box标签"""
        if error_msg is None:
            error_msg = f"未找到<box type='{box_type}'>标签"
        
        box = ssw_card.find('box', {'type': box_type})
        if not box:
            self._add_error(error_msg)
            return None
        return box

    def _validate_box_attributes(self, box, required_attrs: List[str]) -> None:
        """验证box的必要属性"""
        for attr in required_attrs:
            if not box.get(attr):
                self._add_error(f"box缺少{attr}属性")

    def _find_tool_content_by_id(self, tool_content: List[Dict], id_field: str, box_value: Any) -> Optional[Dict]:
        """根据ID字段查找工具内容"""
        if not tool_content:
            return None
        
        for tc in tool_content:
            if str(tc.get(id_field)) == str(box_value):
                return tc
        return None

    def _validate_final_checks(self, model_output: str) -> None:
        """执行最终的通用检查"""
        self._check_forbidden_tags(model_output, "输出")
    
    def remove_xml_tags(self, xml_content: str, preserve_newlines: bool = True) -> str:
        """
        移除XML/HTML标签，返回纯文本内容
        
        Args:
            xml_content (str): 包含XML标签的文本内容
            preserve_newlines (bool): 是否保留换行符，默认为True
            
        Returns:
            str: 移除XML标签后的纯文本内容
        """
        if not xml_content or not isinstance(xml_content, str):
            return ""
        
        try:
            # 使用BeautifulSoup解析XML/HTML内容
            soup = BeautifulSoup(xml_content, 'html.parser')
            
            # 提取纯文本内容
            plain_text = soup.get_text()
            
            if preserve_newlines:
                # 保留换行符，但清理多余的空白
                lines = plain_text.split('\n')
                # 移除每行首尾空白，但保留换行结构
                cleaned_lines = [line.strip() for line in lines]
                # 移除连续的空行，只保留单个空行
                result_lines = []
                prev_empty = False
                for line in cleaned_lines:
                    if line == "":
                        if not prev_empty:
                            result_lines.append(line)
                        prev_empty = True
                    else:
                        result_lines.append(line)
                        prev_empty = False
                
                return '\n'.join(result_lines).strip()
            else:
                # 不保留换行符，将所有空白字符规范化为单个空格
                return ' '.join(plain_text.split()).strip()
                
        except Exception as e:
            # 如果BeautifulSoup解析失败，使用正则表达式作为备选方案
            try:
                # 使用正则表达式移除XML/HTML标签
                clean_text = re.sub(r'<[^>]+>', '', xml_content)
                
                if preserve_newlines:
                    # 保留换行符结构
                    lines = clean_text.split('\n')
                    cleaned_lines = [line.strip() for line in lines]
                    # 移除连续的空行
                    result_lines = []
                    prev_empty = False
                    for line in cleaned_lines:
                        if line == "":
                            if not prev_empty:
                                result_lines.append(line)
                            prev_empty = True
                        else:
                            result_lines.append(line)
                            prev_empty = False
                    
                    return '\n'.join(result_lines).strip()
                else:
                    return ' '.join(clean_text.split()).strip()
                    
            except Exception as regex_error:
                # 如果正则表达式也失败，返回原始内容（不太可能发生）
                return xml_content.strip()
    
    def _check_forbidden_tags(self, content: str, context: str = "") -> None:
        """检查禁止的XML标签"""
        for tag in self.FORBIDDEN_TAGS:
            if re.search(re.escape(tag), content, re.IGNORECASE):
                self._add_error(f"{context}包含禁止的XML标签: {tag}")
    
    def _check_no_links_in_print(self, content: str, context: str = "") -> None:
        """检查print标签内容中不应包含链接"""
        # 检查HTTP/HTTPS链接
        http_pattern = r'https?://[^\s<>"\']+'
        if re.search(http_pattern, content, re.IGNORECASE):
            self._add_error(f"{context}的<print>内容不应包含HTTP/HTTPS链接")
        
        # 检查其他常见链接格式
        link_patterns = [
            r'www\.[^\s<>"\']+'  # www.开头的链接
        ]
        
        for pattern in link_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self._add_error(f"{context}的<print>内容不应包含链接")
        
        # 检查代码块 - 不应包含```包裹的代码块
        code_block_pattern = r'```.*?```'
        if re.search(code_block_pattern, content, re.DOTALL):
            self._add_error(f"{context}的<print>内容不应包含```包裹的代码块")
    
    def _parse_xml(self, content: str) -> Optional[BeautifulSoup]:
        """解析XML内容"""
        try:
            return BeautifulSoup(content, 'xml')
        except Exception as e:
            self._add_error(f"XML解析失败: {str(e)}")
            return None
    
    def _check_ssw_card_structure(self, model_output: str) -> Optional[BeautifulSoup]:
        """检查基本的ssw-card结构"""
        if not re.search(r'<ssw-card>.*?</ssw-card>', model_output, re.DOTALL):
            if self.tool_call_name != 'dictionary_search':
                self._add_error("未找到顶级<ssw-card>标签")
                return None
            else:
                return None
        
        soup = self._parse_xml(model_output)
        if not soup:
            return None
            
        ssw_card = soup.find('ssw-card')
        if not ssw_card:
            self._add_error("无法解析<ssw-card>标签")
            return None
        return soup
    
    def check_card_by_type(self, tool_call_name: str, model_output: str, tool_content) -> Dict[str, Any]:
        """根据工具类型检查卡片内容"""
        self.tool_call_name = tool_call_name
        self.tool_content = tool_content
        self.model_output = model_output
        self._reset_validation()
        if tool_call_name == None and 'ssw-card' in model_output:
            self._add_error("未调用工具但返回了卡片")
            self.validation_result['output_type'] = 'ssw-card'
            return self.validation_result
        elif tool_call_name != None and (tool_content == '未搜索到相关内容' or tool_content == None or tool_content == '') and 'ssw-card' in model_output:
            self._add_error("调用工具无返回内容但返回了卡片")
            self.validation_result['output_type'] = 'ssw-card'
            return self.validation_result
        elif tool_call_name == None and (tool_content == '未搜索到相关内容' or tool_content == None or tool_content == '') and 'ssw-card' not in model_output:
            self._check_no_links_in_print(model_output,'普通文本')
            self._check_forbidden_tags(model_output,'普通文本')
            self.validation_result['output_type'] = 'text'
            return self.validation_result
        elif tool_call_name != None and (tool_content == '未搜索到相关内容' or tool_content == None or tool_content == '') and 'ssw-card' not in model_output:
            self._check_no_links_in_print(model_output,'普通文本')
            self._check_forbidden_tags(model_output,'普通文本')
            self.validation_result['output_type'] = 'text'
            return self.validation_result
        else:
            # 映射工具名称到验证函数
            self.validation_result['output_type'] = 'ssw-card'
            validators = {
                'generate_tbl_question': self.tongbu_lian,
                'search_tbl_question': self.tongbu_lian,
                'recite_search': self.beisong,
                'query_en_word': self.query_en_word,
                'query_knowledge_card': self.knowledge_card,
                'dictionary_search': self.dictionary_search,
                'chinese_dictation': self.dictation_chinese,
                'english_dictation': self.dictation_english,
                'exampaper_search': self.exampaper_search,
            }
            # 验证format输出
            validator = validators.get(tool_call_name)
            if not validator:
                self._add_error(f"未知的工具类型: {tool_call_name}")
                return self.validation_result
            
            return validator(model_output, tool_content)

    def tongbu_lian(self, model_output: str, tool_content) -> Dict[str, Any]:
        """验证同步练习题输出格式"""
        self.validation_result["extracted"] = {
            "box_content": None,
            "exercise_count": 0,
            "ids_mapping": {}
        }
        
        soup = self._check_ssw_card_structure(model_output)
        if not soup:
            return self.validation_result
        
        ssw_card = soup.find('ssw-card')
        box = ssw_card.find('box', attrs={'type': 'exercise-child-box', 'content-type': '练习题'})
        
        # 1. 检查前置回复
        self._check_print_tag_content(ssw_card, "前置回复")
        
        # 2. 检查box模块
        box = self._find_and_validate_box(ssw_card, 'exercise-child-box', "未找到<box type='exercise-child-box'>标签")
        if not box:
            return self.validation_result
        
        # 3. 检查知识点标题格式
        box_print = box.find('print')
        if not box_print:
            self._add_error("box内缺少<print>标签")
            return self.validation_result
        
        box_content = box_print.text.strip()
        self.validation_result["extracted"]["box_content"] = box_content
        
        # 检查box内print内容中不应包含链接
        self._check_no_links_in_print(box_content, "box内容")
        
        # 4. 验证固定标题行
        if "-- 学而思｜同步练" not in box_content:
            self._add_error("缺少固定标题行: '<span style=\"color:gray\">-- 学而思｜同步练</span>'")
        
        # 5. 验证题目编号连续性
        exercise_items = re.findall(r'(\d+)\.\s*(.*?)(?=\s*\n\s*\d+\.|\Z)', box_content, re.DOTALL)
        if not exercise_items:
            self._add_error("未检测到编号题目 (格式要求: 1.content1, 2.content2)")
            return self.validation_result
        
        self.validation_result["extracted"]["exercise_count"] = len(exercise_items)
        
        # 6. 验证ids属性
        try:
            # 从box.next_sibling中正则匹配字典
            pattern = r'</box ids[^>]*>'
            box_end = re.findall(pattern, model_output)[0]
            ids_data = re.search(r'\{.*?\}', box_end).group()
            ids_data = json.loads(ids_data.replace('\\','').replace("'",'"'))
            if not isinstance(ids_data, dict):
                raise ValueError("ids属性必须是字典格式")
        except:
            self._add_error("ids属性不是有效的JSON字典")
            return self.validation_result
        else:
            try:
                # 检查ID数量匹配
                if len(ids_data) != len(exercise_items):
                    self._add_error(f"ids属性数量({len(ids_data)})与题目数量({len(exercise_items)})不匹配")
                
                # 检查编号连续性 (1,2,3...)
                expected_keys = [str(i) for i in range(1, len(exercise_items)+1)]
                if list(ids_data.keys()) != expected_keys:
                    self._add_error(f"ids键值应为连续数字 {expected_keys}，实际为 {list(ids_data.keys())}")
                
                self.validation_result["extracted"]["ids_mapping"] = ids_data
            except (json.JSONDecodeError, ValueError) as e:
                self._add_error(f"ids属性不是有效的JSON字典: {str(e)}")
        
        # 7. 检查内容完整性 (如有原始题目)
        que_id_list=[]
        for idx, tc in enumerate(tool_content):
            que_id_list.append(tc.get('queId', ''))
        if tool_content:
            missing_queids = []
            for ids in list(ids_data.values()):
                if ids not in que_id_list:
                    self._add_error(f"queId与ids属性不匹配: {ids} not in {que_id_list}")
        
        # 8. 最终检查
        self._validate_final_checks(model_output)
        return self.validation_result

    def beisong(self, model_output: str, tool_content) -> Dict[str, Any]:
        """验证背诵内容输出格式"""
        self.validation_result["extracted"] = {
            "title": None,
            "author": None,
            "content": None,
            "poem_id": None
        }
        
        soup = self._check_ssw_card_structure(model_output)
        if not soup:
            return self.validation_result
        
        ssw_card = soup.find('ssw-card')
        
        # 检查前置回复中不应包含链接
        self._check_print_tag_content(ssw_card, "前置回复")
        
        # 1. 检查box模块
        box = self._find_and_validate_box(ssw_card, 'beisong')
        if not box:
            return self.validation_result
        
        title = box.find('title')
        author = box.find('author')
        content = box.find('content')
        
        if not title or not author or not content:
            self._add_error("未找到<title>、<author>或<content>标签")
            return self.validation_result
        
        # 提取内容用于调试
        self.validation_result["extracted"]["title"] = title.text.strip() if title.text else ""
        self.validation_result["extracted"]["author"] = author.text.strip() if author.text else ""
        self.validation_result["extracted"]["content"] = content.text.strip() if content.text else ""
        self.validation_result["extracted"]["poem_id"] = box.get('pomeId')
        # 从tool content中选出模型挑选的那一个
        matched_content = self._find_tool_content_by_id(tool_content, 'id', self.validation_result["extracted"]["poem_id"])
        tool_content_match = matched_content is not None
        # 验证与原始查询的匹配性
        if tool_content_match:
            # if title.text.strip() != tool_content.get('title', '').replace(' ', ''):
            #     self._add_error(f"标题不匹配，期望: {tool_content.get('title', '')}, 实际: {title.text.strip()}")
            # if author.text.strip() != tool_content.get('author', ''):
            #     self._add_error(f"作者不匹配，期望: {tool_content.get('author', '')}, 实际: {author.text.strip()}")
            if box.get('pomeId') != str(matched_content.get('id', '')):
                self._add_error(f"pomeId不匹配，期望: {matched_content.get('id', '')}, 实际: {box.get('pomeId')}")
        
        # 检查内容中的禁止标签
        self._check_forbidden_tags(content.text if content.text else "", "content标签")
        
        # 最终检查
        self._validate_final_checks(model_output)
        return self.validation_result

    def knowledge_card(self, model_output: str, tool_content: Optional[Dict] = None) -> Dict[str, Any]:
        """验证知识卡片输出格式"""
        self.validation_result["extracted"] = {
            "image_source": None,
            "print_count": 0,
            "has_image": False
        }
        
        soup = self._check_ssw_card_structure(model_output)
        if not soup:
            return self.validation_result
        
        ssw_card = soup.find('ssw-card')
        
        # 1. 检查是否包含<box>标签（知识卡片不应该有box）
        box = ssw_card.find('box')
        if box:
            self._add_error("知识卡片不应包含<box>标签")
        
        # 2. 检查<print>标签（可选，位置灵活）
        print_tags = ssw_card.find_all('print')
        self.validation_result["extracted"]["print_count"] = len(print_tags)
        
        # 如果存在<print>标签，检查其内容不包含禁止的XML标签和链接
        for i, print_tag in enumerate(print_tags, 1):
            print_content = print_tag.text if print_tag.text else ""
            self._check_forbidden_tags(print_content, f"第{i}个<print>标签内容")
            self._check_no_links_in_print(print_content, f"第{i}个<print>标签")
        
        # 3. 检查<Image>标签（必须存在）
        image_tag = ssw_card.find('Image')
        if not image_tag:
            self._add_error("未找到<Image>标签")
        else:
            self.validation_result["extracted"]["has_image"] = True
            # 检查source属性
            source_attr = image_tag.get('source')
            if not source_attr:
                self._add_error("<Image>标签缺少source属性")
            else:
                self.validation_result["extracted"]["image_source"] = source_attr
                
                # 如果提供了tool_content，验证image_url是否匹配
                for tc in tool_content:
                    if tc and 'image_url' in tc:
                        if source_attr != tc['image_url']:
                            self._add_error(f"Image source不匹配，期望: {tool_content}, 实际: {source_attr}")
            
        # 4. 检查整体内容中的禁止XML标签（除了<print>内容已检查的部分）
        temp_output = model_output
        for print_tag in print_tags:
            if print_tag.text:
                temp_output = temp_output.replace(print_tag.text, "")
        
        self._check_forbidden_tags(temp_output, "输出的非<print>部分")
        
        return self.validation_result
    
    def query_en_word(self, model_output: str, tool_content: Optional[Dict] = None) -> Dict[str, Any]:
        """验证英语单词查询输出格式"""
        self.validation_result["extracted"] = {
            "text_content": model_output.strip(),
            "has_xml_tags": False,
            "text_length": len(model_output.strip())
        }
        
        # 1. 检查不应包含<ssw-card>标签
        if re.search(r'<ssw-card>.*?</ssw-card>', model_output, re.DOTALL):
            self._add_error("输出不应包含<ssw-card>样式内容，应为普通文本")
        
        # 2. 检查不应包含任何XML/HTML标签
        if re.search(r'<[^>]+>', model_output):
            self._add_error("输出应为纯文本，不应包含任何XML/HTML标签")
            self.validation_result["extracted"]["has_xml_tags"] = True
        
        # 3. 检查输出不为空
        if not model_output.strip():
            self._add_error("输出内容不能为空")
        
        return self.validation_result

    def dictation_chinese(self, model_output: str, tool_content: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """验证中文听写输出格式"""
        self.validation_result["extracted"] = {
            "box_content": None,
            "dictation_types": [],
            "dictation_count": 0,
            "field_content_check": {}
        }
        
        soup = self._check_ssw_card_structure(model_output)
        if not soup:
            return self.validation_result
        
        ssw_card = soup.find('ssw-card')
        
        # 检查前置回复中不应包含链接
        self._check_print_tag_content(ssw_card, "前置回复")
        
        # 1. 检查box模块
        box = self._find_and_validate_box(ssw_card, 'dictation')
        if not box:
            return self.validation_result
        
        # 2. 检查必要的属性
        required_attrs = ['subject_type', 'grade_id', 'version_id', 'term_id']
        self._validate_box_attributes(box, required_attrs)
        
        # 3. 验证与tool_content的匹配性
        matched_content = self._find_tool_content_by_id(tool_content, 'text_id', box.get('text_id'))
        if matched_content:
            # 检查基本信息匹配
            if box.get('subject_type') != matched_content.get('subject_type'):
                self._add_error("subject_type不匹配")
            if box.get('grade_id') != str(matched_content.get('grade_id')):
                self._add_error("grade_id不匹配")
        else:
            self._add_error(f"text_id匹配失败: {box.get('text_id')}")
            return self.validation_result
        # 4. 检查听写内容
        box_print = box.find('print')
        if box_print:
            box_content = box_print.text.strip()
            self.validation_result["extracted"]["box_content"] = box_content
            
            # 检查box内print内容中不应包含链接
            self._check_no_links_in_print(box_content, "box内容")
            
            # 检查是否包含预期的听写类型
            dictation_types = ['会认', '会写', '读读写写', '词语']
            found_types = []
            for dt in dictation_types:
                if dt in box_content:
                    found_types.append(dt)
            
            self.validation_result["extracted"]["dictation_types"] = found_types
            if not found_types:
                self._add_error("未找到预期的听写类型（会认、会写、读读写写、词语）")
            
            # 5. 检查字段内容提取的准确性
            if matched_content:
                content_data = matched_content
                
                # 字段映射：显示名称 -> 数据字段名称
                field_mapping = {
                    '会认': 'huiren',
                    '会写': 'huixie', 
                    '读读写写': 'duduxiexie',
                    '词语': 'ciyu'
                }
                
                for display_name, field_name in field_mapping.items():
                    if display_name in box_content and field_name in content_data:
                        # 获取原始数组数据
                        field_array = content_data.get(field_name, [])
                        if isinstance(field_array, list) and field_array:
                            # 用英文逗号拼接数组元素
                            expected_content = ','.join(field_array)
                            
                            # 记录检查结果用于调试
                            self.validation_result["extracted"]["field_content_check"][display_name] = {
                                "field_name": field_name,
                                "original_array": field_array,
                                "expected_content": expected_content,
                                "found_in_output": expected_content in box_content
                            }
                            
                            # 检查拼接后的内容是否在输出中
                            if expected_content not in box_content:
                                self._add_error(f"'{display_name}'字段内容不完整，期望包含: {expected_content}")
                        else:
                            self.validation_result["extracted"]["field_content_check"][display_name] = {
                                "field_name": field_name,
                                "original_array": field_array,
                                "error": "字段数据为空或格式不正确"
                            }
                            self._add_error(f"'{display_name}'字段数据为空或格式不正确")
        
        # 6. 最终检查
        self._validate_final_checks(model_output)
        
        return self.validation_result
    
    def dictation_english(self, model_output: str, tool_content: Optional[Dict] = None) -> Dict[str, Any]:
        """验证英语听写输出格式"""
        self.validation_result["extracted"] = {
            "box_content": None,
            "word_count": 0,
            "words": [],
            "title_check": False,
            "content_check": {}
        }
        
        soup = self._check_ssw_card_structure(model_output)
        if not soup:
            return self.validation_result
        
        ssw_card = soup.find('ssw-card')
        
        # 检查前置回复中不应包含链接
        self._check_print_tag_content(ssw_card, "前置回复")
        
        # 1. 检查box模块
        box = self._find_and_validate_box(ssw_card, 'dictation')
        if not box:
            return self.validation_result
        
        # 2. 检查必要的属性（包括英语听写特有的text_id）
        required_attrs = ['subject_type', 'grade_id', 'version_id', 'term_id', 'text_id']
        self._validate_box_attributes(box, required_attrs)
        
        # 3. 验证与tool_content的匹配性
        matched_content = self._find_tool_content_by_id(tool_content, 'text_id', box.get('text_id'))
        if matched_content:
            if box.get('subject_type') != matched_content.get('subject_type'):
                self._add_error("subject_type不匹配")
            if box.get('grade_id') != str(matched_content.get('grade_id')):
                self._add_error("grade_id不匹配")
            if box.get('version_id') != str(matched_content.get('version_id')):
                self._add_error("version_id不匹配")
            if box.get('term_id') != str(matched_content.get('term_id')):
                self._add_error("term_id不匹配")
            if box.get('text_id') != str(matched_content.get('text_id')):
                self._add_error("text_id不匹配")
        else:
            self._add_error(f"text_id匹配失败: {box.get('text_id')}")
            return self.validation_result
        # 4. 检查title标签
        title = box.find('title')
        if not title:
            self._add_error("未找到<title>标签")
        elif title.text.strip() != '单词':
            self._add_error(f"title内容错误，期望'单词'，实际'{title.text.strip()}'")
        else:
            self.validation_result["extracted"]["title_check"] = True
        
        # 5. 检查content标签和danci字段内容
        content = box.find('content')
        if not content:
            self._add_error("未找到<content>标签")
        else:
            content_text = content.text.strip()
            self.validation_result["extracted"]["box_content"] = content_text
            
            # 检查danci字段内容提取的准确性
            if matched_content and 'danci' in matched_content:
                danci_array = matched_content.get('danci', [])
                if isinstance(danci_array, list) and danci_array:
                    # 用英文逗号拼接数组元素
                    expected_content = ','.join(danci_array)
                    
                    # 记录检查结果用于调试
                    self.validation_result["extracted"]["content_check"] = {
                        "original_array": danci_array,
                        "expected_content": expected_content,
                        "actual_content": content_text,
                        "found_in_output": expected_content == content_text
                    }
                    
                    self.validation_result["extracted"]["words"] = danci_array
                    self.validation_result["extracted"]["word_count"] = len(danci_array)
                    
                    # 检查拼接后的内容是否与输出完全匹配
                    if expected_content != content_text:
                        self._add_error(f"content内容不匹配，期望: '{expected_content}'，实际: '{content_text}'")
                else:
                    self.validation_result["extracted"]["content_check"] = {
                        "original_array": danci_array,
                        "error": "danci字段数据为空或格式不正确"
                    }
                    self._add_error("danci字段数据为空或格式不正确")
        
        # 6. 最终检查
        self._validate_final_checks(model_output)
        
        return self.validation_result

    def exampaper_search(self, model_output: str, tool_content: Optional[Dict] = None) -> Dict[str, Any]:
        """验证试卷搜索输出格式"""
        self.validation_result["extracted"] = {
            "keywords": None,
            "bank_count": 0,
            "web_count": 0,
            "bank_contents": [],
            "web_contents": [],
            "paper_sources": []
        }
        
        soup = self._check_ssw_card_structure(model_output)
        if not soup:
            return self.validation_result
        
        ssw_card = soup.find('ssw-card')
        
        # 检查前置回复中不应包含链接
        self._check_print_tag_content(ssw_card, "前置回复")
        
        # 1. 检查box模块
        box = self._find_and_validate_box(ssw_card, 'exam-paper')
        if not box:
            return self.validation_result
        
        # 2. 检查<keywords>标签
        keywords_tag = box.find('keywords')
        if not keywords_tag:
            self._add_error("未找到<keywords>标签")
        else:
            keywords_content = keywords_tag.get('content')
            if not keywords_content:
                self._add_error("<keywords>标签缺少content属性")
            else:
                self.validation_result["extracted"]["keywords"] = keywords_content
        
        # 3. 检查<bank>和<web>标签
        bank_tags = box.find_all('bank')
        web_tags = box.find_all('web')
        
        self.validation_result["extracted"]["bank_count"] = len(bank_tags)
        self.validation_result["extracted"]["web_count"] = len(web_tags)
        
        # 验证bank标签的content属性
        for i, bank_tag in enumerate(bank_tags, 1):
            bank_content = bank_tag.get('content')
            if not bank_content:
                self._add_error(f"第{i}个<bank>标签缺少content属性")
            else:
                self.validation_result["extracted"]["bank_contents"].append(bank_content)
                # 检查content是否为有效的JSON字符串
                try:
                    json.loads(bank_content)
                except json.JSONDecodeError:
                    self._add_error(f"第{i}个<bank>标签的content不是有效的JSON字符串")
        
        # 验证web标签的content属性
        for i, web_tag in enumerate(web_tags, 1):
            web_content = web_tag.get('content')
            if not web_content:
                self._add_error(f"第{i}个<web>标签缺少content属性")
            else:
                self.validation_result["extracted"]["web_contents"].append(web_content)
                # 检查content是否为有效的JSON字符串
                try:
                    json.loads(web_content)
                except json.JSONDecodeError:
                    self._add_error(f"第{i}个<web>标签的content不是有效的JSON字符串")
        
        # 4. 验证与tool_content的匹配性
        if tool_content:
            # 检查keywords匹配
            if 'keyword' in tool_content and keywords_tag:
                expected_keywords = tool_content['keyword']
                actual_keywords = keywords_tag.get('content')
                if expected_keywords != actual_keywords:
                    self._add_error(f"keywords不匹配，期望: {expected_keywords}, 实际: {actual_keywords}")
            
            # 检查试卷列表
            if 'paper_list' in tool_content:
                paper_list = tool_content['paper_list']
                
                # 按来源分类获取所有可能的content
                bank_papers = [p for p in paper_list if p.get('source') == 'bank']
                web_papers = [p for p in paper_list if p.get('source') == 'web']
                
                # 获取所有可能的bank和web内容
                available_bank_contents = [p.get('content', '') for p in bank_papers]
                available_web_contents = [p.get('content', '') for p in web_papers]
                
                self.validation_result["extracted"]["paper_sources"] = [p.get('source') for p in paper_list]
                
                # 检查输出中的每个bank标签的content是否在tool_content的bank来源中存在
                for i, bank_content in enumerate(self.validation_result["extracted"]["bank_contents"], 1):
                    if bank_content not in available_bank_contents:
                        self._add_error(f"第{i}个<bank>标签的content在tool_content的bank来源中不存在")
                
                # 检查输出中的每个web标签的content是否在tool_content的web来源中存在
                for i, web_content in enumerate(self.validation_result["extracted"]["web_contents"], 1):
                    if web_content not in available_web_contents:
                        self._add_error(f"第{i}个<web>标签的content在tool_content的web来源中不存在")
                # 检查是否输出了超出tool_content范围的bank数量
                if len(bank_tags) > len(bank_papers):
                    self._add_error(f"<bank>标签数量({len(bank_tags)})超过了tool_content中bank来源试卷数量({len(bank_papers)})")
                
                # 检查是否输出了超出tool_content范围的web数量
                if len(web_tags) > len(web_papers):
                    self._add_error(f"<web>标签数量({len(web_tags)})超过了tool_content中web来源试卷数量({len(web_papers)})")
        
        # 5. 检查是否至少有一个bank或web标签
        if len(bank_tags) == 0 and len(web_tags) == 0:
            self._add_error("至少需要一个<bank>或<web>标签")
        
        # 6. 最终检查
        self._validate_final_checks(model_output)
        
        return self.validation_result
    
    def dictionary_search(self, model_output: str, tool_content: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """验证汉语字典查询输出格式"""
        self.validation_result["extracted"] = {
            "character_count": 0,
            "characters": [],
        }
        
        soup = self._check_ssw_card_structure(model_output)
        if not soup:
            return self.validation_result
        
        ssw_card = soup.find('ssw-card')
        
        # 检查前置回复中不应包含链接
        self._check_print_tag_content(ssw_card, "前置回复")
        # 检查<character_svg type="words" content="{name字段的内容}" svg_url="SVG链接地址"/> //content是具体的字和词  
        character_svg_tag = ssw_card.find('character_svg')
        if character_svg_tag:
            content = character_svg_tag.get('content')
            if content:
                self.validation_result["extracted"]["svg_content"] = content
            # 检查content和tool_content中name字段的内容是否一致
            for tc in tool_content:
                if tc['name'] != content:
                    self._add_error(f"content和tool_content中name字段的内容不一致，期望: {tc['name']}, 实际: {content}")
                    break
        # 3. 最终检查
        self._validate_final_checks(model_output)
        
        return self.validation_result
