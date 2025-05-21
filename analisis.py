# %%
import ply.lex as lex
import ply.yacc as yacc
from arbol import (
    BreakStatement,
    CaseClause,
    DefaultClause,
    Literal,
    BinaryOp,
    SwitchStatement,
    Visitor,
    Variable,
    UnaryOp,
    WhileStatement,
    DoWhileStatement,
    ForStatement,
    IfStatement,
    Block,
    Assignment,
    Declaration,
    Program,
)

literals = ["+", "-", "*", "/", "%", "(", ")", "{", "}", ";", "=", ":"]
reserved = {
    "while": "WHILE",
    "if": "IF",
    "else": "ELSE",
    "int": "INT",
    "bool": "BOOL",
    "float": "FLOAT",
    "char": "CHAR",
    "main": "MAIN",
    "do": "DO",
    "for": "FOR",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "break": "BREAK",
}
tokens = [
    "INTLIT",
    "ID",
    "OR",
    "AND",
    "LEQ",
    "GEQ",
    "LT",
    "GT",
    "EQ",
    "NEQ",
    "MINUS",
    "PLUS",
    "NOT",
    "COLON",
] + list(reserved.values())

t_ignore = " \t"


def t_ID(t):
    r"[a-zA-Z_][a-zA-Z_0-9]*"
    t.type = reserved.get(t.value, "ID")
    return t


def t_INTLIT(t):
    r"[0-9]+"
    t.value = int(t.value)
    return t


def t_newline(t):
    r"\n+"
    t.lexer.lineno += len(t.value)


t_OR = r"\|\|"
t_AND = r"&&"
t_LT = r"<"
t_LEQ = r"<="
t_GT = r">"
t_GEQ = r">="
t_EQ = r"=="
t_NEQ = r"!="
t_MINUS = r"-"
t_NOT = r"!"
t_PLUS = r"\+"
t_COLON = r":"


def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)


# %%

precedence = (
    ("nonassoc", "ELSE"),
    ("right", "NOT", "UMINUS"),
    ("left", "OR"),
    ("left", "AND"),
    ("nonassoc", "EQ", "NEQ"),
    ("nonassoc", "LT", "LEQ", "GT", "GEQ"),
    ("left", "PLUS", "MINUS"),
    ("left", "*", "/", "%"),
)


def p_empty(p):
    """
    empty :
    """
    p[0] = []


def p_program(p):
    """
    program : INT MAIN "(" ")" "{" declarations statements "}"
    """
    p[0] = Program(p[6], p[7])


def p_declarations(p):
    """
    declarations : declaration declarations
                 | empty
    """
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = []


def p_declaration(p):
    """
    declaration : type ID ";"
    """
    p[0] = Declaration(p[1], p[2])


def p_type(p):
    """
    type : INT
         | BOOL
         | FLOAT
         | CHAR
    """
    p[0] = p.slice[1].type


def p_statements(p):
    """
    statements : statement statements
               | empty
    """
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = []


def p_statement(p):
    """
    statement : ';'
              | block
              | assignment
              | if_statement
              | while_statement
              | do_while_statement
              | for_statement
              | switch_statement
              | break_statement
    """
    p[0] = p[1]


def p_block(p):
    """
    block : '{' statements '}'
    """
    p[0] = Block(p[2])


def p_assignment_no_semi(p):
    """
    assignment_no_semi : ID '=' expression
    """
    p[0] = Assignment(p[1], p[3])


def p_assignment(p):
    """
    assignment : assignment_no_semi ';'
    """
    p[0] = p[1]


def p_assignment_opt(p):
    """
    assignment_opt : assignment_no_semi
                   | empty
    """
    p[0] = p[1]


def p_if_statement(p):
    """
    if_statement : IF '(' expression ')' statement ELSE statement
                    | IF '(' expression ')' statement
    """
    if len(p) == 8:
        p[0] = IfStatement(p[3], p[5], p[7])
    else:
        p[0] = IfStatement(p[3], p[5], None)


def p_for_statement(p):
    """
    for_statement : FOR '(' assignment_opt ';' expression_opt ';' assignment_opt ')' statement
    """
    init = p[3]
    cond = p[5]
    post = p[7]
    body = p[9]
    p[0] = ForStatement(init, cond, post, body)


def p_do_while_statement(p):
    """
    do_while_statement : DO statement WHILE '(' expression ')' ';'
    """
    p[0] = DoWhileStatement(p[5], p[2])


def p_while_statement(p):
    """
    while_statement : WHILE '(' expression ')' statement
    """

    p[0] = WhileStatement(p[3], p[5])


def p_switch_statement(p):
    """
    switch_statement : SWITCH '(' expression ')' '{' case_clauses default_clause_opt '}'
    """
    p[0] = SwitchStatement(p[3], p[6], p[7])


def p_case_clauses(p):
    """
    case_clauses : case_clause case_clauses
                 | empty
    """
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = []


def p_case_clause(p):
    """
    case_clause : CASE INTLIT COLON statements
    """
    p[0] = CaseClause(p[2], p[4])


def p_default_clause_opt(p):
    """
    default_clause_opt : DEFAULT COLON statements
                       | empty
    """
    if len(p) == 4:
        p[0] = DefaultClause(p[3])
    else:
        p[0] = None


def p_break_statement(p):
    "break_statement : BREAK ';'"
    p[0] = BreakStatement()


def p_expression_opt(p):
    """
    expression_opt : expression
                   | empty
    """
    p[0] = p[1]


def p_expression(p):
    """
    expression : expression OR conjunction
               | conjunction
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinaryOp(p[2], p[1], p[3])


def p_conjunction(p):
    """
    conjunction : conjunction AND equality
                | equality
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinaryOp(p[2], p[1], p[3])


def p_equality(p):
    """
    equality : relation equ_op relation
             | relation
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinaryOp(p[2], p[1], p[3])


def p_equ_op(p):
    """
    equ_op : EQ
           | NEQ
    """
    p[0] = p[1]


def p_relation(p):
    """
    relation : addition rel_op addition
             | addition
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinaryOp(p[2], p[1], p[3])


def p_rel_op(p):
    """
    rel_op : LT
           | LEQ
           | GT
           | GEQ
    """
    p[0] = p[1]


def p_addition(p):
    """
    addition : addition add_op term
             | term
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinaryOp(p[2], p[1], p[3])


def p_add_op(p):
    """
    add_op : PLUS
           | MINUS
    """
    p[0] = p[1]


def p_term(p):
    """
    term : term mul_op factor
         | factor
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinaryOp(p[2], p[1], p[3])


def p_mul_op(p):
    """
    mul_op : '*'
           | '/'
           | '%'
    """
    p[0] = p[1]


def p_factor(p):
    """
    factor : unary_op factor
           | primary
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = UnaryOp(p[1], p[2])


def p_unary_op(p):
    """unary_op : MINUS %prec UMINUS
    | NOT   %prec NOT"""
    p[0] = p[1]


def p_primary(p):
    """
    primary : INTLIT
            | ID
            | '(' expression ')'
    """
    if len(p) == 2:
        if isinstance(p[1], int):
            p[0] = Literal(p[1], "INT")
        else:
            p[0] = Variable(p[1])
    else:
        p[0] = p[2]


def p_error(p):
    print(f"Syntax error at '{p.value}'")


# %%
data = """
int main() {
    int x;
    int y;
    int i;
    bool flag;
    flag = -flag;
    flag = !flag ;
    y = 2 - 1; 
    x = 0;
    if (x < 3) { x = x + 1; } else { x = x - 1; }
    while (x > 0) { x = x - 1; }
    do { x = x - 1; } while (x > 0);
    for (i = 0; i < 10; i = i + 1) {x = x + i;}
    switch (x) {
        case 0: x = x + 1;
        case 2: x = x + 2;
        default: x = x + 3;
    }
}
"""
lexer = lex.lex()
parser = yacc.yacc(start="program")
ast = parser.parse(data)

# %%
from llvmlite import ir

intType = ir.IntType(32)
module = ir.Module(name="prog")

fnty = ir.FunctionType(intType, [])
func = ir.Function(module, fnty, name="main")

entry = func.append_basic_block("entry")
builder = ir.IRBuilder(entry)


class IRGenerator(Visitor):
    def __init__(self):
        self.stack = []
        self.symbols = {}
        self.typemap = {
            "INT": ir.IntType(32),
            "BOOL": ir.IntType(1),
            "FLOAT": ir.DoubleType(),
            "CHAR": ir.IntType(8),
        }

    def visit_program(self, node: Program):
        for decl in node.declarations:
            decl.accept(self)

        for stmt in node.statements:
            stmt.accept(self)

        if not builder.block.is_terminated:
            builder.ret(ir.Constant(intType, 0))

    def visit_declaration(self, node: Declaration):
        ty = self.typemap[node.typ]
        with builder.goto_entry_block():
            alloca = builder.alloca(ty, name=node.identifier)
        self.symbols[node.identifier] = alloca

        if isinstance(ty, ir.DoubleType):
            zero = ir.Constant(ty, 0.0)
        else:
            zero = ir.Constant(ty, 0)
        builder.store(zero, alloca)

    def visit_block(self, node: Block):
        for stmt in node.statements:
            stmt.accept(self)

    def visit_assignment(self, node: Assignment):
        node.expression.accept(self)
        val = self.stack.pop()
        ptr = self.symbols[node.identifier]
        builder.store(val, ptr)

    def visit_if_statement(self, node: IfStatement):
        node.condition.accept(self)
        cond = self.stack.pop()

        then_bb = func.append_basic_block("if.then")
        else_bb = func.append_basic_block("if.else") if node.else_stmt else None
        merge_bb = func.append_basic_block("if.end")

        if node.else_stmt:
            builder.cbranch(cond, then_bb, else_bb)
        else:
            builder.cbranch(cond, then_bb, merge_bb)

        builder.position_at_start(then_bb)
        node.then_stmt.accept(self)
        builder.branch(merge_bb)

        if node.else_stmt:
            builder.position_at_start(else_bb)
            node.else_stmt.accept(self)
            builder.branch(merge_bb)

        builder.position_at_start(merge_bb)

    def visit_while_statement(self, node: WhileStatement):
        whileHead = func.append_basic_block("while-head")
        whileBody = func.append_basic_block("while-body")
        whileExit = func.append_basic_block("while-exit")
        builder.branch(whileHead)
        builder.position_at_start(whileHead)
        node.condition.accept(self)
        condition = self.stack.pop()
        builder.cbranch(condition, whileBody, whileExit)
        builder.position_at_start(whileBody)
        node.statement.accept(self)
        builder.branch(whileHead)
        builder.position_at_start(whileExit)

    def visit_do_while_statement(self, node: DoWhileStatement):
        doBody = func.append_basic_block("do.body")
        doCond = func.append_basic_block("do.cond")
        doExit = func.append_basic_block("do.exit")
        builder.branch(doBody)
        builder.position_at_start(doBody)
        node.body.accept(self)
        builder.branch(doCond)
        builder.position_at_start(doCond)
        node.condition.accept(self)
        condVal = self.stack.pop()
        builder.cbranch(condVal, doBody, doExit)
        builder.position_at_start(doExit)

    def visit_for_statement(self, node: ForStatement):
        entryBB = builder.block
        if node.init:
            node.init.accept(self)

        condBB = func.append_basic_block("for.cond")
        bodyBB = func.append_basic_block("for.body")
        postBB = func.append_basic_block("for.post")
        exitBB = func.append_basic_block("for.exit")
        builder.branch(condBB)
        builder.position_at_start(condBB)
        if node.condition:
            node.condition.accept(self)
            condVal = self.stack.pop()
        else:
            condVal = ir.Constant(ir.IntType(1), 1)
        builder.cbranch(condVal, bodyBB, exitBB)
        builder.position_at_start(bodyBB)
        node.body.accept(self)
        builder.branch(postBB)
        builder.position_at_start(postBB)
        if node.update:
            node.update.accept(self)
        builder.branch(condBB)
        builder.position_at_start(exitBB)

    def visit_switch_statement(self, node):
        # Create the end block for the switch statement
        end_block = func.append_basic_block("switch.end")

        # Evaluate the switch expression and store its value
        node.expr.accept(self)
        switch_val = self.stack.pop()

        # Create basic blocks for each case and the default case
        case_blocks = {}
        for case in node.cases:
            case_blocks[case.value] = func.append_basic_block(f"case_{case.value}")
        default_block = (
            func.append_basic_block("default") if node.default else end_block
        )

        # Create the switch instruction
        switch_inst = builder.switch(switch_val, default_block)
        for value, block in case_blocks.items():
            switch_inst.add_case(ir.Constant(switch_val.type, value), block)

        # Generate code for each case block
        for i, case in enumerate(node.cases):
            builder.position_at_start(case_blocks[case.value])
            self.current_break_target = end_block
            self.visit_case_statements(case.stmts)

            # If the block is not terminated, branch to the next case or the end block
            if not builder.block.is_terminated:
                next_block = (
                    case_blocks[node.cases[i + 1].value]
                    if i + 1 < len(node.cases)
                    else end_block
                )
                builder.branch(next_block)

        # Generate code for the default block, if it exists
        if node.default:
            builder.position_at_start(default_block)
            self.current_break_target = end_block
            self.visit_case_statements(node.default.stmts)
            if not builder.block.is_terminated:
                builder.branch(end_block)

        builder.position_at_start(end_block)

    def visit_case_statements(self, statements):
        for stmt in statements:
            stmt.accept(self)

    def visit_break_statement(self, node):
        if self.current_break_target is None:
            raise Exception("Out of context for break statement")
        builder.branch(self.current_break_target)

    def visit_variable(self, node):
        if node.name not in self.symbols:
            raise NameError(f"Undeclared variable '{node.name}'")
        ptr = self.symbols[node.name]
        self.stack.append(builder.load(ptr))

    def visit_literal(self, node: Literal):
        self.stack.append(ir.Constant(intType, node.value))

    def visit_unary_op(self, node: UnaryOp):
        node.operand.accept(self)
        val = self.stack.pop()

        if node.op == "-":
            zero = ir.Constant(val.type, 0)
            self.stack.append(builder.sub(zero, val))
        elif node.op == "!":
            zero = ir.Constant(val.type, 0)
            self.stack.append(builder.icmp_signed("==", val, zero))
        else:
            raise ValueError(f"UnaryOp desconocido {node.op}")

    def visit_binary_op(self, node: BinaryOp) -> None:
        node.lhs.accept(self)
        node.rhs.accept(self)
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        if node.op == "+":
            self.stack.append(builder.add(lhs, rhs))
        elif node.op == "-":
            self.stack.append(builder.sub(lhs, rhs))
        elif node.op == "*":
            self.stack.append(builder.mul(lhs, rhs))
        elif node.op == "/":
            self.stack.append(builder.sdiv(lhs, rhs))
        elif node.op == "%":
            self.stack.append(builder.srem(lhs, rhs))
        elif node.op == "<":
            result = builder.icmp_signed("<", lhs, rhs)
            self.stack.append(result)
        elif node.op == "<=":
            result = builder.icmp_signed("<=", lhs, rhs)
            self.stack.append(result)
        elif node.op == ">":
            result = builder.icmp_signed(">", lhs, rhs)
            self.stack.append(result)
        elif node.op == ">=":
            result = builder.icmp_signed(">=", lhs, rhs)
            self.stack.append(result)
        elif node.op == "==":
            result = builder.icmp_signed("==", lhs, rhs)
            self.stack.append(result)
        elif node.op == "!=":
            result = builder.icmp_signed("!=", lhs, rhs)
            self.stack.append(result)
        elif node.op == "||":
            result = builder.or_(lhs, rhs)
            self.stack.append(result)
        elif node.op == "&&":
            result = builder.and_(lhs, rhs)
            self.stack.append(result)
        else:
            raise ValueError(f"Operador desconocido {node.op}")


# %%
ast = parser.parse(data)
visitor = IRGenerator()
ast.accept(visitor)
# print(module)
# %%
import runtime as rt
from ctypes import CFUNCTYPE, c_int

engine = rt.create_execution_engine()
mod = rt.compile_ir(engine, str(module))
func_ptr = engine.get_function_address("main")

# Run the function via ctypes
cfunc = CFUNCTYPE(c_int)(func_ptr)
res = cfunc()
print("main() =", res)
print(mod)


# %%
