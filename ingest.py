import argparse
import os.path

from inverted_index import Index


def get_index_from_cmd_arg(data_in):
    index = Index()
    if data_in.endswith('.txt'):
        index.index_document(data_in)
    elif os.path.isdir(data_in):
        index.index_directory(data_in)
    elif os.path.isfile(data_in):
        index.load_from_file(data_in)
    else:
        raise IOError(f'could not create index from {data_in}')
    return index


def pretty_print(results):
    if not results:
        print('  no matches returned')
        return
    left_pad = max(len(doc) for doc in results) + 2
    print(f'  {"document":>{left_pad}}  score')
    for doc, score in results.items():
        print(f'  {doc:>{left_pad}}  {score:.4}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_in', type=str,
                        help='path to input data (can be a .txt file, a '
                             'directory, or a saved index)')
    parser.add_argument('-s', '--save-to', type=str,
                        help='if specified, save index to the given filepath')
    args = parser.parse_args()

    index = get_index_from_cmd_arg(args.data_in)
    if args.save_to:
        index.save_to_file(args.save_to)

    index.display()

    while True:
        query = input('query: ')
        results = index.query_bm25(query)
        pretty_print(results)


if __name__ == '__main__':
    main()
