"""Reply bank for E7 (adds DRIVE) by the same context distillation as E2."""

import sys
import e2_data, env7

e2_data.env = env7
if __name__ == "__main__":
    e2_data.main(e2_data.argparse.Namespace(model=sys.argv[1], out=sys.argv[2], variants=2, batch=48))
