# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 Martin Sandve Alnæs
#
# This file is part of DIJITSO.
#
# DIJITSO is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DIJITSO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DIJITSO. If not, see <http://www.gnu.org/licenses/>.

"""This file contains the commands available through command-line dijitso-cache.

Each function cmd_<cmdname> becomes a subcommand invoked by::

    dijitso-cache cmdname ...args

The docstrings in the cmd_<cmdname> are shown when running::

    dijitso-cache cmdname --help

The 'args' argument to cmd_* is a Namespace object with the commandline arguments.

"""

import os
import re

from dijitso import __version__
from dijitso.cache import glob_cache, grep_cache, clean_cache
from dijitso.cache import extract_lib_signatures
from dijitso.cache import extract_files, extract_function
from dijitso.system import read_textfile


def parse_categories(categories):
    if categories == "all":
        return ("inc", "src", "lib", "log")
    return categories.split(",")


def args_version(parser):
    pass


def cmd_version(args, params):
    "print dijitso version"
    print(__version__)


def args_config(parser):
    parser.add_argument("--key", default="", help="specific key to show (e.g. build.cxxflags)")


def cmd_config(args, params):
    "show configuration"
    # Show single value if asked for
    key = args.key
    if key:
        name = key
        value = params
        for k in key.split("."):
            value = value[k]
        # Compiler flags etc are more useful in space separated form:
        if isinstance(value, tuple):
            value = " ".join(value)
        print("    %s: %s" % (name, value))
        return 0

    # Pick non-empty categories
    categories = sorted(c for c in params if params[c])
    print("Showing default flags for dijitso:")
    for category in categories:
        print("%s:" % (category,))
        for name in sorted(params[category]):
            value = params[category][name]
            # Compiler flags etc are more useful in space separated form:
            if isinstance(value, tuple):
                value = " ".join(value)
            print("    %s: %s" % (name, value))
    return 0


def args_show(parser):
    parser.add_argument("--categories", default="all",
                        help="comma separated list to enable file types (inc,src,lib,log)")
    parser.add_argument("--no-summary", action="store_true",
                        help="don't show summary")
    parser.add_argument("--files", action="store_true",
                        help="show file lists")
    parser.add_argument("--signatures", action="store_true",
                        help="show library signatures")


def cmd_show(args, params):
    "show lists of files in cache"
    cache_params = params["cache"]

    summary = not args.no_summary
    files = args.files
    signatures = args.signatures
    categories = parse_categories(args.categories)

    gc = glob_cache(cache_params, categories=categories)

    if signatures:
        sigs = extract_lib_signatures(cache_params)
        print("\n".join("\t" + s for s in sorted(sigs)))
    if files:
        for cat in categories:
            print("\n".join("\t" + f for f in sorted(gc.get(cat, ()))))
    if summary:
        print("dijitso cache summary (number of cached files):")
        for cat in categories:
            print("%s: %d" % (cat, len(gc.get(cat, ()))))
        # TODO: Add summary of file sizes
    return 0


def args_clean(parser):
    parser.add_argument("--categories", default="inc,src,lib,log",
                        help="comma separated list to enable file types (inc,src,lib,log)")


def cmd_clean(args, params):
    "remove files from cache"
    cache_params = params["cache"]

    dryrun = args.dry_run
    categories = parse_categories(args.categories)

    clean_cache(cache_params, dryrun=dryrun, categories=categories)
    return 0


def args_grep(parser):
    parser.add_argument("--categories", default="inc,src",
                        help="comma separated list to enable file types (inc,src,lib,log)")
    parser.add_argument("--pattern", default="",
                        help="line search pattern")
    parser.add_argument("--regexmode", action="store_true",
                        help="pattern is a regular expression (python style)")
    parser.add_argument("--linenumbers", action="store_true",
                        help="show linenumbers on matches")
    parser.add_argument("--countonly", action="store_true",
                        help="show only match line count for each file")
    parser.add_argument("--filesonly", action="store_true",
                        help="show only filenames with matches")
    parser.add_argument("--signature", default="",
                        help="look for module with this signature (default all)")


def cmd_grep(args, params):
    "grep content of header and source file(s) in cache"
    cache_params = params["cache"]

    # Get command-line arguments
    pattern = args.pattern
    signature = args.signature
    regexmode = args.regexmode
    linenumbers = args.linenumbers
    countonly = args.countonly
    filesonly = args.filesonly
    categories = parse_categories(args.categories)

    if not regexmode:
        pattern = ".*(" + pattern + ").*"
    regex = re.compile(pattern)
    allmatches = grep_cache(regex, cache_params,
                            linenumbers=linenumbers, countonly=countonly,
                            signature=signature,
                            categories=categories)
    if filesonly:
        print("\n".join(sorted(allmatches)))
    elif countonly:
        print("\n".join("%s: %d" % (k, v) for k, v in sorted(allmatches.items())))
    else:
        for fn in sorted(allmatches):
            print("\nFile '%s' matches:" % (fn,))
            if linenumbers:
                print("\n".join("%5d:\t%s" % line for line in allmatches[fn]))
            else:
                print("\n".join("\t" + line for line in allmatches[fn]))
    return 0


def args_grepfunction(parser):
    parser.add_argument("--categories", default="src",
                        help="comma separated list to enable file types (inc,src,lib,log)")
    parser.add_argument("--name", default="",
                        help="function name to search for")
    parser.add_argument("--signature", default="",
                        help="look for module with this signature (default all)")
    parser.add_argument("--no-body", action="store_true",
                        help="don't show function bodies")


def cmd_grepfunction(args, params):
    "search for function name in source files in cache"
    cache_params = params["cache"]

    name = args.name
    signature = args.signature
    categories = parse_categories(args.categories)
    no_body = args.no_body

    pattern = r".*(" + name + r")[ ]*\((.*)"
    regex = re.compile(pattern)

    allmatches = grep_cache(regex, cache_params,
                            linenumbers=True, countonly=False,
                            signature=signature,
                            categories=categories)
    for fn in sorted(allmatches):
        if no_body:
            # Just print signature lines
            print("\nFile '%s' matches:" % (fn,))
            for i, line in allmatches[fn]:
                print("%5d: %s" % (i, line))
        else:
            # Print function bodies
            content = read_textfile(fn)
            lines = content.splitlines() if content else ()
            for i, line in allmatches[fn]:
                print("%s:%d" % (fn, i))
                assert name in lines[i]
                print(extract_function(lines[i:]))
                print()
    return 0


def args_checkout(parser):
    parser.add_argument("--categories", default="inc,src,lib,log",
                        help="comma separated list to enable file types (inc,src,lib,log)")
    parser.add_argument("--signature",
                        help="module signature (required)")


def cmd_checkout(args, params):
    "copy files from cache to a directory"
    cache_params = params["cache"]

    signature = args.signature
    categories = parse_categories(args.categories)

    prefix = "jitcheckout-"
    path = os.curdir
    path = extract_files(signature, cache_params,
                         prefix=prefix, path=path,
                         categories=categories)
    print("Extracted files to '%s'." % (path,))
    return 0
