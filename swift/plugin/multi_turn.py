def check_math_result_and_give_tips(inputs):
    from .orm import MathAccuracy
    acc = MathAccuracy()
    # a trick
    prompt = 'But wait... It seems I made a mistake,'
    contents = [input['messages'][-1]['content'] for input in inputs]
    rewards = acc(contents, [input['solution'] for input in inputs])
    for reward, input in zip(rewards, inputs):
        content = input['messages'][-1]['content']
        if reward < 1 and prompt not in content:
            if '<answer>' in content:
                content = content[:content.index('<answer>')]
            if '</think>' in content:
                content = content[:content.index('</think>')]
            content += prompt
            input['messages'][-1]['content'] = content
            input['finished'] = False
        else:
            input['finished'] = True
    return inputs


def check_math_result_and_give_tips_multi_turn(inputs):
    from .orm import MathAccuracy
    acc = MathAccuracy()
    prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'
    contents = [input['messages'][-1]['content'] for input in inputs]
    rewards = acc(contents, [input['solution'] for input in inputs])
    for reward, input in zip(rewards, inputs):
        content = input['messages'][-2]['content']
        if reward < 1 and prompt not in content:
            input['messages'].append({'role': 'user', 'content': prompt})
            input['finished'] = False
        else:
            input['finished'] = True
    return inputs


multi_turns = {
    'math_tip_trick': check_math_result_and_give_tips,
    'math_tip_trick_multi_turn': check_math_result_and_give_tips_multi_turn,
}
