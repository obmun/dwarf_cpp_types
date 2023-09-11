import logging
import sys

from dwarf_cppdecl.extractor import TypesExtractor

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    FORMAT = "%(levelname)s:%(name)s:%(funcName)s - %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    if len(sys.argv) < 2:
        print('Expected usage: {0} <executable>'.format(sys.argv[0]))
        sys.exit(1)

    # Associated types will be automatically brought in
    types_whitelist = [
        ['TopStruct'],
        # Example syntax: ['root_namespace', 'nested_ns', 'Type']
    ]
    extractor = TypesExtractor()
    extractor.process(sys.argv[1], types_whitelist)
    structs = extractor.scope_tree.flatten()
    pass