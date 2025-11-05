from swift.llm import TemplateType

if __name__ == '__main__':
    template_name_list = TemplateType.get_template_name_list()
    tn_gen = ', '.join([tn for tn in template_name_list if 'generation' in tn])
    tn_chat = ', '.join([tn for tn in template_name_list if 'generation' not in tn])
    print(f'Text Generation: {tn_gen}')
    print(f'Chat: {tn_chat}')
