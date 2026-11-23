"""先运行本文件，生成所有模型共同使用的13变量数据和时间划分。"""
from pathlib import Path
import argparse
from config import DEFAULT_DATA, DEFAULT_RESULTS
from data_protocol import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--year', type=int, default=None, help='仅在有来源依据时修正年份；新CSV默认保留年份')
    args = parser.parse_args()
    b = prepare(args.data, args.results, args.year)
    print(f'13 predictors; {len(b.protocol["train_ids"])} training targets; {len(b.protocol["test_ids"])} test targets.')
    if b.protocol['author_confirmed_year_correction']:
        print('Applied author-confirmed 2023 -> 2017 correction to the supplied CSV copy.')


if __name__ == '__main__':
    main()
