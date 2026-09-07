# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Package containing task implementations."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# Skip helper-only packages when auto-importing task registrations.
_BLACKLIST_PKGS = ["utils", ".mdp", "base", "_registration"]

# Import all task packages in this package.
import_packages(__name__, _BLACKLIST_PKGS)
