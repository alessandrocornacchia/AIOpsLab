import os
import re

# Map between os env variables and str representation
IMP_FLAGS = {
  'IMP_FLAG_REMOVE_ANSI': 'remove_ansi',
  'IMP_FLAG_REDUCE_CONTEXT': 'reduce_context',
}

def read_improvement_flags():
    imp_flags = []

    for env_var in os.environ:
        flag = IMP_FLAGS.get(env_var, None)
        if flag:
            print(f"=== IMPROVEMENT FLAG FOUND: {flag}")
            imp_flags.append(flag)
    return imp_flags


ACTIVE_IMP_FLAGS = read_improvement_flags()

def escape_ansi(line):
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)