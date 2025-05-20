from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ASTNode(ABC):
    @abstractmethod
    def accept(self, visitor: Visitor) -> None:
        pass


class Literal(ASTNode):
    def __init__(self, value: Any, type: str) -> None:
        self.value = value
        self.type = type

    def accept(self, visitor: Visitor):
        visitor.visit_literal(self)


class Variable(ASTNode):
    def __init__(self, name: Any, type: str) -> None:
        self.name = name
        self.type = type

    def accept(self, visitor: Visitor):
        visitor.visit_variable(self)


class BinaryOp(ASTNode):
    def __init__(self, op: str, lhs: ASTNode, rhs: ASTNode) -> None:
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def accept(self, visitor: Visitor):
        visitor.visit_binary_op(self)


class Visitor(ABC):
    @abstractmethod
    def visit_literal(self, node: Literal) -> None:
        pass

    @abstractmethod
    def visit_variable(self, node: Variable) -> None:
        pass

    @abstractmethod
    def visit_binary_op(self, node: BinaryOp) -> None:
        pass


class Calculator(Visitor):
    def __init__(self):
        self.stack = []

    def visit_variable(self, node):
        raise

    def visit_literal(self, node: Literal) -> None:
        self.stack.append(node.value)

    def visit_binary_op(self, node: BinaryOp) -> None:
        node.lhs.accept(self)
        node.rhs.accept(self)
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        if node.op == "+":
            self.stack.append(lhs + rhs)
        elif node.op == "-":
            self.stack.append(lhs - rhs)
        elif node.op == "*":
            self.stack.append(lhs * rhs)
        elif node.op == "/":
            self.stack.append(lhs / rhs)
        elif node.op == "%":
            self.stack.append(lhs % rhs)
        elif node.op == "<":
            self.stack.append(lhs < rhs)
        elif node.op == "<=":
            self.stack.append(lhs <= rhs)
        elif node.op == ">":
            self.stack.append(lhs > rhs)
        elif node.op == ">=":
            self.stack.append(lhs >= rhs)
        elif node.op == "==":
            self.stack.append(lhs == rhs)
        elif node.op == "!=":
            self.stack.append(lhs != rhs)
        elif node.op == "||":
            self.stack.append(lhs or rhs)
        elif node.op == "&&":
            self.stack.append(lhs and rhs)


class Variable:
    def __init__(self, name):
        self.name = name

    def accept(self, visitor):
        visitor.visit_variable(self)


class UnaryOp:
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def accept(self, visitor):
        visitor.visit_unary_op(self)


class WhileStatement:
    def __init__(self, condition, statement):
        self.condition = condition
        self.statement = statement

    def accept(self, visitor):
        visitor.visit_while_statement(self)


class DoWhileStatement:
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def accept(self, visitor):
        return visitor.visit_do_while_statement(self)


class ForStatement:
    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def accept(self, visitor):
        return visitor.visit_for_statement(self)


class IfStatement:
    def __init__(self, condition, then_stmt, else_stmt=None):
        self.condition = condition
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt

    def accept(self, visitor):
        visitor.visit_if_statement(self)


class Block:
    def __init__(self, statements):
        self.statements = statements

    def accept(self, visitor):
        visitor.visit_block(self)


class Assignment:
    def __init__(self, identifier, expression):
        self.identifier = identifier
        self.expression = expression

    def accept(self, visitor):
        visitor.visit_assignment(self)


class Declaration:
    def __init__(self, typ, identifier):
        self.typ = typ
        self.identifier = identifier

    def accept(self, visitor):
        visitor.visit_declaration(self)


class Program:
    def __init__(self, declarations, statements):
        self.declarations = declarations
        self.statements = statements

    def accept(self, visitor):
        visitor.visit_program(self)
