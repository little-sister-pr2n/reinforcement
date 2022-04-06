import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--testing", default="pr1n")
args = parser.parse_args()
print(args.testing)