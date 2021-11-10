import argparse
import re
import socket


def is_ip(ip):
    return re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",ip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Deepspeed Hostfile')
    parser.add_argument('-c', '--cobalt_hostfile', type=str, required=True)
    parser.add_argument('-d', '--deepspeed_hostfile', type=str, required=True)
    parser.add_argument('-s', '--slots', default=8)

    args = parser.parse_args()

    with open(args.cobalt_hostfile) as ch:
        with open(args.deepspeed_hostfile, 'w') as dh:
            for line in ch.readlines():
                host = line.strip()
                if not is_ip(host):
                    ips = socket.gethostbyname_ex(host)[2]
                    ips = [ip for ip in ips if not ip.startswith('127.0.')]
                    host = ips[0]
                dh.write(f'{host} slots={args.slots}\n')
