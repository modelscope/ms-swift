from swift.llm.template import split_str_parts_by


def test_split_str_parts_by():
    print(split_str_parts_by('aaaAction:bb\nbAction Inputs:\nabbb', ['Action:', 'Action Inputs:'], regex_mode=False))
    print(split_str_parts_by('aaaAction:bb\nbAction Inputs:\nabbb', ['Action:', 'Action Inputs:'], regex_mode=True))
    print(split_str_parts_by('aaa<tool_call>bbb</tool_call>ccc', ['<tool_call>.+?</tool_call>'], regex_mode=True))
    print(split_str_parts_by('aaa<image>\nbb\nb<audio>\nabbb', ['<image>', '<audio>', '<video>'], regex_mode=False))
    print(split_str_parts_by('aaa<image>\nbb\nb<audio>\nabbb', ['<image>', '<audio>', '<video>'], regex_mode=True))


if __name__ == '__main__':
    test_split_str_parts_by()
