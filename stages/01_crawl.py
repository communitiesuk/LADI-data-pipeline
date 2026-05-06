"""
Stage 1: Crawl LA websites and extract document text using crawldocs.
"""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description='Crawl LA websites and extract document URLs + text')
    p.add_argument('--config', default='config/pipeline.yaml')
    p.add_argument('--urls', default=None, help='Override URLs file from config')
    p.add_argument('--sample', type=int, default=None, help='Limit to N LAs (for testing)')
    args = p.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)['crawl']

    cmd = [
        sys.executable, '-m', 'crawler_cli',
        '--urls', args.urls or cfg['urls_file'],
        '--name-col', cfg['name_col'],
        '--url-col', cfg['url_col'],
        '--output-dir', cfg['output_dir'],
        '--state-dir', cfg['state_dir'],
        '--aiohttp-concurrency', str(cfg.get('aiohttp_concurrency', 750)),
        '--playwright-tabs', str(cfg.get('playwright_tabs', 30)),
    ]
    if cfg.get('extract_text'):
        cmd.append('--extract-text')
        cmd += ['--extraction-workers', str(cfg.get('extraction_concurrency', 300))]
    if args.sample:
        cmd += ['--sample', str(args.sample)]

    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
