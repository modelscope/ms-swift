import os

if int(os.environ.get('UNSLOTH_PATCH_TRL', '0')) != 0:
    import unlsoth
