# %%
import ply.lex as lex
import ply.yacc as yacc
from llvmlite import ir
import runtime as rt
from ctypes import CFUNCTYPE, c_int

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
    FunctionDecl,
    CallExpr,
    CallStatement,
    ReturnStatement,
    Cast
)

# %%
literals = ["*", "/", "%", "(", ")", "{", "}", ";", "=", ":", ","]
reserved = {
    "while": "WHILE", "if": "IF", "else": "ELSE",
    "int": "INT", "bool": "BOOL", "float": "FLOAT", "char": "CHAR",
    "main": "MAIN", "do": "DO", "for": "FOR",
    "const": "CONST",
    "switch": "SWITCH", "case": "CASE", "default": "DEFAULT",
    "break": "BREAK", "return": "RETURN",
}
tokens = [
    "FLOATLIT", "INTLIT", "ID", "STRINGLIT",
    "OR", "AND", "LEQ", "GEQ", "LT", "GT", "EQ", "NEQ",
    "MINUS", "PLUS", "NOT", "COLON", "INC", "DEC","PLUSEQ","DECEQ"
] + list(reserved.values())

t_ignore = " \t"
t_ignore_COMMENT = r"//.*"  # ignora comentarios de línea

def t_STRINGLIT(t):
    r'"([^"\\]|\\.)*"'
    t.value = t.value[1:-1]  # elimina comillas
    return t

def t_ID(t):
    r"[a-zA-Z_][a-zA-Z_0-9]*"
    t.type = reserved.get(t.value, "ID")
    return t


def t_FLOATLIT(t):
    r"([0-9]+\.[0-9]*|\.[0-9]+)([fF])?"
    txt = t.value
    if txt[-1] in ("f","F"):
        txt = txt[:-1]
    t.value = float(txt)
    return t


def t_INTLIT(t):
    r"[0-9]+"
    t.value = int(t.value)
    return t


def t_newline(t):
    r"\n+"
    t.lexer.lineno += len(t.value)

t_OR    = r"\|\|"
t_AND   = r"&&"
t_LT    = r"<"
t_LEQ   = r"<="
t_GT    = r">"
t_GEQ   = r">="
t_EQ    = r"=="
t_NEQ   = r"!="
t_INC = r"\+\+"
t_PLUSEQ = r"\+="
t_DEC    = r"--"
t_DECEQ = r"\-="
t_MINUS = r"-"
t_NOT   = r"!"
t_PLUS  = r"\+"
t_COLON = r":"


def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)



# %%

precedence = (
    ("nonassoc", "ELSE"),
    ("right", "NOT", "UMINUS", "INC", "DEC"),
    ("left", "OR"),
    ("left", "AND"),
    ("nonassoc", "EQ", "NEQ"),
    ("nonassoc", "LT", "LEQ", "GT", "GEQ"),
    ("left", "PLUSEQ", "DECEQ"),  
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
    program : functions
    """
    p[0] = Program(p[1])


def p_functions(p):
    """
    functions : mainfunction functions
              | function functions
              | function_prototype functions
              | empty
    """
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = []


def p_return_statement(p):
    """
    return_statement : RETURN expression_opt ";"
    """
    p[0] = ReturnStatement(p[2])

def p_function_prototype(p):
    """
    function_prototype : type ID "(" parameters ")" ";"
    """
    decl = FunctionDecl(p[1], p[2], p[4], [], [])
    decl.is_proto = True
    p[0] = decl

def p_function(p):
    """
    function : type ID "(" parameters ")" "{" declarations statements "}"
    """
    p[0] = FunctionDecl(p[1], p[2], p[4], p[7], p[8])


def p_mainfunction(p):
    """
    mainfunction : INT MAIN "(" ")" "{" declarations statements "}"
    """
    p[0] = FunctionDecl("INT", "main", [], p[6], p[7])


def p_call_statement(p):
    """
    call_statement : ID "(" arguments ")" ";"
    """
    p[0] = CallStatement(p[1], p[3])


def p_arguments(p):
    """
    arguments : expression_list
              | empty
    """
    p[0] = p[1]


def p_expression_list(p):
    """
    expression_list : expression
                    | expression "," expression_list
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]


def p_parameters(p):
    """
    parameters : parameter_list
               | empty
    """
    p[0] = p[1]


def p_parameter_list(p):
    """
    parameter_list : parameter
                   | parameter "," parameter_list
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]


def p_parameter(p):
    """
    parameter : type ID
    """
    p[0] = (p[1], p[2])


def p_declarations(p):
    """
    declarations : declaration declarations
                 | empty
    """
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = []

def p_decl_no_semi(p):
    """
    decl_no_semi : type ID
                  | type ID '=' expression
    """
    init = p[4] if len(p) == 5 else None
    p[0] = Declaration(p[1], p[2], init)

def p_declaration(p):
    """
    declaration : type ID ";"
                | type ID '=' expression ';'
                | CONST type ID ';'
                | CONST type ID '=' expression ';'
    """
    if p[1] == "const":
        #  CONST type ID    ...
        typ      = p[2]
        ident    = p[3]
        init     = p[5] if len(p) == 6 else None
    else:
        #  type ID ...
        typ      = p[1]
        ident    = p[2]
        init     = p[4] if len(p) == 5 else None

    p[0] = Declaration(typ, ident, init)


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
              | declaration
              | assignment
              | if_statement
              | while_statement
              | do_while_statement
              | for_statement
              | return_statement
              | call_statement
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
                      | ID PLUSEQ expression
                      | ID DECEQ expression
                      | ID INC
                      | ID DEC
    """
    var = Variable(p[1])
    if p[2] == '=':
        rhs = p[3]
    elif p[2] == '+=':
        rhs = BinaryOp('+', var, p[3])
    elif p[2] == '-=':
        rhs = BinaryOp('-', var, p[3])
    elif p[2] == '++':
        rhs = BinaryOp('+', var, Literal(1, "INT"))
    else:  # '--'
        rhs = BinaryOp('-', var, Literal(1, "INT"))
    p[0] = Assignment(p[1], rhs)


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
    for_statement : FOR '(' decl_no_semi ';' expression_opt ';' assignment_opt ')' statement
                  | FOR '(' assignment_opt ';' expression_opt ';' assignment_opt ')' statement
    """
    p[0] = ForStatement(p[3], p[5], p[7], p[9])


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
           | INC factor
           | DEC factor
           | primary
           | postfix
    """
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 3 and p[1] in ('++', '--'):
        name = p[2].name
        op = '+' if p[1] == '++' else '-'
        p[0] = Assignment(name, BinaryOp(op, p[2], Literal(1, "INT")))
    else:
        p[0] = UnaryOp(p[1], p[2])


def p_postfix(p):
    """
    postfix : primary INC
            | primary DEC
    """
    name = p[1].name  # asumimos que primary → Variable
    op = '+' if p[2] == '++' else '-'
    p[0] = Assignment(name, BinaryOp(op, p[1], Literal(1, "INT")))

def p_unary_op(p):
    """unary_op : MINUS %prec UMINUS
                | NOT   %prec NOT"""
    p[0] = p[1]


def p_primary(p):
    """
    primary : '(' type ')' primary
            | ID '(' arguments ')'
            | INTLIT
            | FLOATLIT
            | STRINGLIT
            | ID
            | '(' expression ')'
    """
    if len(p) == 5 and isinstance(p[2], str) and p[2] in ("INT","FLOAT","BOOL","CHAR"):
        p[0] = Cast(p[2], p[4])
    elif len(p) == 5 and p.slice[1].type == "ID":
        p[0] = CallExpr(p[1], p[3])
    elif len(p) == 2:
        tok = p.slice[1].type
        if tok == "INTLIT":
            p[0] = Literal(p[1], "INT")
        elif tok == "FLOATLIT":
            p[0] = Literal(p[1], "FLOAT")
        elif tok == "STRINGLIT":
            p[0] = Literal(p[1], "STRING")
        else:
            p[0] = Variable(p[1])
    else:
        p[0] = p[2]

def p_error(p):
    print(f"Syntax error at '{p.value}'")


# %%
with open("test.c", 'r', encoding='utf-8') as f:
        data = f.read()
lexer = lex.lex()
parser = yacc.yacc(start="program")
ast = parser.parse(data)

# %%

intType = ir.IntType(32)
module = ir.Module(name="prog")


class IRGenerator(Visitor):
    def __init__(self, module: ir.Module):
        self.module = module
        self.builder: ir.IRBuilder = None
        self.fn: ir.Function = None
        self.symbols = {}
        self.stack = []
        self.typemap = {
            "INT": ir.IntType(32),
            "BOOL": ir.IntType(1),
            "FLOAT": ir.FloatType(),
            "CHAR": ir.IntType(8),
        }
        voidptr = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr], var_arg=True)
        self.printf = ir.Function(module, printf_ty, name="printf")
        self.strings = {}
    
    def _get_str_constant(self, text):
        if text in self.strings:
            return self.strings[text]
        arr = bytearray(text.encode("utf8"))
        const = ir.Constant(ir.ArrayType(ir.IntType(8), len(arr)), arr)
        gv = ir.GlobalVariable(self.module, const.type, name=f".str{len(self.strings)}")
        gv.linkage = "internal"
        gv.global_constant = True
        gv.initializer = const
        self.strings[text] = gv
        return gv
    
    def visit_cast(self, node: Cast):
        # primero baja el valor
        node.expr.accept(self)
        val = self.stack.pop()
        from_ty = val.type
        to_ty = self.typemap[node.target_type]
        if isinstance(from_ty, ir.FloatType) and isinstance(to_ty, ir.IntType):
            casted = self.builder.fptosi(val, to_ty)
        elif isinstance(from_ty, ir.IntType) and isinstance(to_ty, ir.FloatType):
            casted = self.builder.sitofp(val, to_ty)
        elif isinstance(from_ty, ir.DoubleType) and isinstance(to_ty, ir.FloatType):
            casted = self.builder.fptrunc(val, to_ty)
        elif isinstance(from_ty, ir.FloatType) and isinstance(to_ty, ir.DoubleType):
            casted = self.builder.fpext(val, to_ty)
        else:
            casted = val
        self.stack.append(casted)

    def visit_program(self, node: Program):
        for fn in node.functions:
            if getattr(fn, 'is_proto', False):
                param_types = [self.typemap[t] for t,_ in fn.params]
                fn_ty = ir.FunctionType(self.typemap[fn.ret_type], param_types)
                ir.Function(self.module, fn_ty, name=fn.name)
        for fn in node.functions:
            if not getattr(fn, 'is_proto', False):
                self.visit_function_decl(fn)

    def visit_function_decl(self, node: FunctionDecl):
        param_types = [self.typemap[t] for t, _ in node.params]
        fn_ty = ir.FunctionType(self.typemap[node.ret_type], param_types)
        # fn = ir.Function(self.module, fn_ty, name=node.name)
        if node.name in self.module.globals:
            fn = self.module.get_global(node.name)
        else:
            fn = ir.Function(self.module, fn_ty, name=node.name)

        entry = fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self.func = fn

        for i, (_, name) in enumerate(node.params):
            alloca = self.builder.alloca(fn.args[i].type, name=name)
            self.builder.store(fn.args[i], alloca)
            self.symbols[name] = alloca

        for decl in node.declarations:
            decl.accept(self)

        for stmt in node.statements:
            stmt.accept(self)

        if not self.builder.block.is_terminated:
            if node.ret_type == "INT":
                self.builder.ret(ir.Constant(self.typemap["INT"], 0))
            else:
                self.builder.ret(ir.Constant(self.typemap[node.ret_type], 0.0))

    def visit_declaration(self, node: Declaration):
        ty = self.typemap[node.typ]
        alloca = self.builder.alloca(ty, name=node.identifier)
        self.symbols[node.identifier] = alloca
        zero = ir.Constant(ty, 0.0 if isinstance(ty, ir.FloatType) else 0)
        self.builder.store(zero, alloca)
        if node.initializer:
            node.initializer.accept(self)
            val = self.stack.pop()
            self.builder.store(val, alloca)

    def visit_block(self, node: Block):
        for stmt in node.statements:
            stmt.accept(self)

    def visit_assignment(self, node: Assignment):
        node.expression.accept(self)
        val = self.stack.pop()
        ptr = self.symbols[node.identifier]
        self.builder.store(val, ptr)

    def visit_if_statement(self, node: IfStatement):
        node.condition.accept(self)
        cond = self.stack.pop()

        then_bb  = self.func.append_basic_block("if.then")
        merge_bb = self.func.append_basic_block("if.end")

        if node.else_stmt:
            else_bb = self.func.append_basic_block("if.else")
            self.builder.cbranch(cond, then_bb, else_bb)
        else:
            else_bb = merge_bb
            self.builder.cbranch(cond, then_bb, merge_bb)

        self.builder.position_at_start(then_bb)
        node.then_stmt.accept(self)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)

        if node.else_stmt:
            self.builder.position_at_start(else_bb)
            node.else_stmt.accept(self)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

        self.builder.position_at_start(merge_bb)

    def visit_while_statement(self, node: WhileStatement):
        whileHead = self.func.append_basic_block("while-head")
        whileBody = self.func.append_basic_block("while-body")
        whileExit = self.func.append_basic_block("while-exit")
        self.builder.branch(whileHead)
        self.builder.position_at_start(whileHead)
        node.condition.accept(self)
        condition = self.stack.pop()
        self.builder.cbranch(condition, whileBody, whileExit)
        self.builder.position_at_start(whileBody)
        node.statement.accept(self)
        self.builder.branch(whileHead)
        self.builder.position_at_start(whileExit)

    def visit_do_while_statement(self, node: DoWhileStatement):
        body_bb = self.func.append_basic_block("do.body")
        cond_bb = self.func.append_basic_block("do.cond")
        exit_bb = self.func.append_basic_block("do.exit")
        self.builder.branch(body_bb)
        self.builder.position_at_start(body_bb)
        node.body.accept(self)
        if not self.builder.block.is_terminated:
            self.builder.branch(cond_bb)
        self.builder.position_at_start(cond_bb)
        node.condition.accept(self)
        condVal = self.stack.pop()
        self.builder.cbranch(condVal, body_bb, exit_bb)
        self.builder.position_at_start(exit_bb)


    def visit_for_statement(self, node: ForStatement):
        entryBB = self.builder.block
        if node.init:
            node.init.accept(self)

            condBB = self.func.append_basic_block("for.cond")
        bodyBB = self.func.append_basic_block("for.body")
        postBB = self.func.append_basic_block("for.post")
        exitBB = self.func.append_basic_block("for.exit")
        self.builder.branch(condBB)
        self.builder.position_at_start(condBB)
        if node.condition:
            node.condition.accept(self)
            condVal = self.stack.pop()
        else:
            condVal = ir.Constant(ir.IntType(1), 1)
        self.builder.cbranch(condVal, bodyBB, exitBB)
        self.builder.position_at_start(bodyBB)
        node.body.accept(self)
        self.builder.branch(postBB)
        self.builder.position_at_start(postBB)
        if node.update:
            node.update.accept(self)
        self.builder.branch(condBB)
        self.builder.position_at_start(exitBB)

    def visit_switch_statement(self, node):
        # Create the end block for the switch statement
        end_block = self.func.append_basic_block("switch.end")

        # Evaluate the switch expression and store its value
        node.expr.accept(self)
        switch_val = self.stack.pop()

        # Create basic blocks for each case and the default case
        case_blocks = {}
        for case in node.cases:
            case_blocks[case.value] = self.func.append_basic_block(f"case_{case.value}")
        default_block = (
            self.func.append_basic_block("default") if node.default else end_block
        )

        # Create the switch instruction
        switch_inst = self.builder.switch(switch_val, default_block)
        for value, block in case_blocks.items():
            switch_inst.add_case(ir.Constant(switch_val.type, value), block)

        # Generate code for each case block
        for i, case in enumerate(node.cases):
            self.builder.position_at_start(case_blocks[case.value])
            self.current_break_target = end_block
            self.visit_case_statements(case.stmts)

            # If the block is not terminated, branch to the next case, the default block or the end block
            if not self.builder.block.is_terminated:
                if i + 1 < len(node.cases):
                    next_block = case_blocks[node.cases[i + 1].value]
                elif node.default:
                    next_block = default_block
                else:
                    next_block = end_block
                self.builder.branch(next_block)

        # Generate code for the default block, if it exists
        if node.default:
            self.builder.position_at_start(default_block)
            self.current_break_target = end_block
            self.visit_case_statements(node.default.stmts)
            if not self.builder.block.is_terminated:
                self.builder.branch(end_block)

        self.builder.position_at_start(end_block)


    def visit_case_statements(self, statements):
        for stmt in statements:
            stmt.accept(self)

    def visit_break_statement(self, node):
        if self.current_break_target is None:
            raise Exception("Out of context for break statement")
        self.builder.branch(self.current_break_target)

    def visit_variable(self, node: Variable):
        ptr = self.symbols[node.name]
        self.stack.append(self.builder.load(ptr))

    def visit_literal(self, node: Literal):
        const = (
            ir.Constant(self.typemap["FLOAT"], node.value)
            if node.type == "FLOAT"
            else ir.Constant(self.typemap[node.type], node.value)
        )
        self.stack.append(const)

    def visit_unary_op(self, node: UnaryOp):
        node.operand.accept(self)
        v = self.stack.pop()
        if node.op == "-":
            zero = ir.Constant(v.type, 0)
            op = (
                self.builder.fsub
                if isinstance(v.type, ir.FloatType)
                else self.builder.sub
            )
            self.stack.append(op(zero, v))
        else:
            zero = ir.Constant(v.type, 0)
            self.stack.append(self.builder.icmp_signed("==", v, zero))

    def visit_binary_op(self, node: BinaryOp) -> None:
        node.lhs.accept(self)
        node.rhs.accept(self)
        rhs = self.stack.pop()
        lhs = self.stack.pop()

        is_float = isinstance(lhs.type, ir.FloatType)

        if node.op == "+":
            instr = self.builder.fadd if is_float else self.builder.add
            self.stack.append(instr(lhs, rhs))
        elif node.op == "-":
            instr = self.builder.fsub if is_float else self.builder.sub
            self.stack.append(instr(lhs, rhs))
        elif node.op == "*":
            instr = self.builder.fmul if is_float else self.builder.mul
            self.stack.append(instr(lhs, rhs))
        elif node.op == "/":
            instr = self.builder.fdiv if is_float else self.builder.sdiv
            self.stack.append(instr(lhs, rhs))
        elif node.op == "%":
            instr = self.builder.frem if is_float else self.builder.srem
            self.stack.append(instr(lhs, rhs))

        elif node.op in ("<", "<=", ">", ">=", "==", "!="):
            if is_float:
                pred_map = {
                    "<": "ULT",
                    "<=": "ULE",
                    ">": "UGT",
                    ">=": "UGE",
                    "==": "UEQ",
                    "!=": "UNE",
                }
                pred = pred_map[node.op]
                cmp = self.builder.fcmp_ordered(pred, lhs, rhs)
            else:
                cmp = self.builder.icmp_signed(node.op, lhs, rhs)
            self.stack.append(cmp)

        elif node.op == "||":
            self.stack.append(self.builder.or_(lhs, rhs))
        elif node.op == "&&":
            self.stack.append(self.builder.and_(lhs, rhs))

        else:
            raise ValueError(f"Operador desconocido {node.op}")

    def visit_return_statement(self, node: ReturnStatement):
        if node.expression:
            node.expression.accept(self)
            val = self.stack.pop()
            self.builder.ret(val)
        else:
            self.builder.ret(ir.Constant(self.typemap["INT"], 0))

    def visit_call_expr(self, node: CallExpr):
        if node.name == "printf":
            fmt_lit = node.arguments[0]
            assert isinstance(fmt_lit, Literal) and fmt_lit.type == "STRING"
            fmt_text = fmt_lit.value + "\0"
            gv = self._get_str_constant(fmt_text)
            ptr = self.builder.bitcast(gv, ir.IntType(8).as_pointer())
            args = [ptr]
            for arg in node.arguments[1:]:
                arg.accept(self)
                v = self.stack.pop()
                if isinstance(v.type, ir.FloatType):
                    v = self.builder.fpext(v, ir.DoubleType())
                args.append(v)
            call = self.builder.call(self.printf, args)
            self.stack.append(call)
        else:
            args = []
            for arg in node.arguments:
                arg.accept(self)
                args.append(self.stack.pop())
            fn = self.module.get_global(node.name)
            call = self.builder.call(fn, args)
            self.stack.append(call)

    def visit_call_statement(self, node: CallStatement):
        self.visit_call_expr(CallExpr(node.name, node.arguments))


# %%
ast = parser.parse(data)
visitor = IRGenerator(module)
ast.accept(visitor)
print(module)
# %%


engine = rt.create_execution_engine()
mod = rt.compile_ir(engine, str(module))
func_ptr = engine.get_function_address("main")

# Run the function via ctypes
cfunc = CFUNCTYPE(c_int)(func_ptr)
res = cfunc()
print("main() =", res)
print(mod)


# %%
