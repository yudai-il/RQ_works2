# -*- coding: utf-8 -*-

class OptimizerException(Exception):
    pass


class InvalidArgument(OptimizerException):
    pass


class OptimizationFailed(OptimizerException):
    pass


