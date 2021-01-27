import argparse
import os

import codecs

def shard(input_file, output_file_format, bytes_per_shard, max_shards=None):
    if not os.path.exists(input_file):
        raise ValueError('Could not find input file {}'.format(input_file))
    if '{index}' not in output_file_format:
        raise ValueError('output_file_format must contain "{index}"')
    if not os.path.exists(os.path.dirname(output_file_format)):
        os.makedir(os.path.dirname(output_file_format))

    index = 1
    ofile_name = output_file_format.format(index=index)
    ofile = open(ofile_name, 'w', encoding='utf-8')
    with codecs.open(input_file, 'r', encoding='utf-8') as ifile:
        for line in ifile.readlines():
            ofile.write(line)
            if line == '\n' and ofile.tell() > bytes_per_shard:
                index += 1
                ofile.close()
                if max_shards is not None and index > max_shards:
                    return
                ofile_name = output_file_format.format(index=index)
                ofile = open(ofile_name, 'w', encoding='utf-8')
    ofile.close()


def parse_value_as_int(value):
    postfix_map = {'K': 1000, 'M': 1000000, 'B': 1000000000}
    if isinstance(value, int) or isinstance(value, float):
        return int(value)
    if value.isdigit():
        return int(value)
    if len(value) > 1:
        return int(float(value[:-1]) * postfix_map[value[-1].upper()])
    raise ValueError('Unable to parse "{}" as integer'.format(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text file sharder')

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input text file with articles separated by '
                             'blank lines.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory to write shards to')
    parser.add_argument('-f', '--format', type=str, default='shard_{index}.txt',
                        help='Output file name format. {index} will be '
                             'with the numeric index of the output shard')
    parser.add_argument('-b', '--size', type=str, required='100M',
                        help='Maximum number of bytes per shard')
    parser.add_argument('-n', '--max_shards', type=int, default=None,
                        help='Limit maximum number of shards')
    args = parser.parse_args()

    print('Sharding {} to {}'.format(args.input, args.output))

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    output_file_format = os.path.join(args.output, args.format)

    bytes_per_shard = parse_value_as_int(args.size)

    shard(args.input, output_file_format, bytes_per_shard, args.max_shards)

    print('Finished sharding')
