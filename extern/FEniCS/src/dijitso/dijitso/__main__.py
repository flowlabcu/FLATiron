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

"""This is the commandline interface to dijitso. For usage help, run 'dijitso --help'."""

import sys
import argparse

from dijitso.params import validate_params
import dijitso.cmdline as cmd_namespace


def build_commands(cmd_namespace):
    """Collects functions called cmd_<basename> from given namespace.

    Returns dict {basename: function}.
    """
    commands = {}
    cmd_args = {}
    for name in list(cmd_namespace.keys()):
        if name.startswith("cmd_"):
            cmd_name = name.replace("cmd_", "")
            cmd = cmd_namespace[name]
            commands[cmd_name] = cmd
            args_name = "args_" + cmd_name
            if args_name in cmd_namespace:
                cmd_args[cmd_name] = cmd_namespace.get(args_name)
    return commands, cmd_args


def add_top_arguments(parser):
    "Add arguments to top level parser."
    parser.add_argument("--verbose", "-v", default=False,
                        help="set logging level")
    parser.add_argument("--cache-dir", "-r", default=None,
                        help="use non-default cache root path")
    parser.add_argument("--dry-run", "-n", default=False,
                        help="only show what would be done, don't modify filesystem")


def extract_params_from_args(args):
    p = {}
    p["cache"] = {}
    if args.cache_dir is not None:
        p["cache"]["cache_dir"] = args.cache_dir
    return p


def add_common_arguments(parser):
    "Add arguments to each subparser."
    pass


def add_cmd_arguments(cmd, parser, args):
    "Add arguments specific to a command."
    if hasattr(cmd, "add_arguments"):
        cmd.add_arguments(parser)


def build_parsers(commands, args):
    """Builds a top parser with subparsers for each command."""
    top_parser = argparse.ArgumentParser()
    add_top_arguments(top_parser)

    subparsers = top_parser.add_subparsers(help="command description", dest="cmd_name")
    cmd_parsers = {}
    for cmd_name, cmd in commands.items():
        parser = subparsers.add_parser(cmd_name, help=cmd.__doc__)
        add_common_arguments(parser)
        if cmd_name in args:
            args[cmd_name](parser)
        cmd_parsers[cmd_name] = parser

    return top_parser, subparsers, cmd_parsers


def main(args=None):
    """This is the commandline tool for the python module dijitso."""

    if args is None:
        args = sys.argv[1:]

    # Build subparsers for each command
    commands, cmd_args = build_commands(vars(cmd_namespace))
    top_parser, subparsers, cmd_parsers = build_parsers(commands, cmd_args)

    # Populate args namespace
    args_ns = argparse.Namespace()
    top_parser.parse_args(args, namespace=args_ns)

    # Extract generic params
    params = extract_params_from_args(args_ns)
    params = validate_params(params)

    # Run the chosen command (argparse doesn't allow
    # getting to this point with an invalid cmd_name)
    assert args_ns.cmd_name in commands
    cmd = commands[args_ns.cmd_name]
    return cmd(args_ns, params)


if __name__ == "__main__":
    sys.exit(main())
