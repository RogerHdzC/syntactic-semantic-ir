# %%
import ply.lex as lex
import ply.yacc as yacc
from arbol import Literal, BinaryOp, Visitor, Variable, UnaryOp, WhileStatement, IfStatement, Block, Assignment, Declaration, Program

literals = ['+','-','*','/', '%', '(', ')', '{', '}', ';', '=']
reserved = {
    'while': 'WHILE',
    'if':    'IF',
    'else':  'ELSE',
    'int':   'INT',
    'bool':  'BOOL',
    'float': 'FLOAT',
    'char':  'CHAR',
    'main':  'MAIN',
}
tokens = [
    'INTLIT', 'ID', 'OR', 'AND', 'LEQ', 'GEQ', 'LT', 'GT',
    'EQ', 'NEQ', 'MINUS', 'PLUS', 'NOT',
] + list(reserved.values())

t_ignore  = ' \t'

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')
    return t

def t_INTLIT(t):
    r'[0-9]+'
    t.value = int(t.value)
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

t_OR = r'\|\|'
t_AND = r'&&'
t_LT = r'<'
t_LEQ = r'<='
t_GT = r'>'
t_GEQ = r'>='
t_EQ = r'=='
t_NEQ = r'!='
t_MINUS = r'-'
t_NOT = r'!'
t_PLUS = r'\+'

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

# %%

precedence = (
    ('nonassoc', 'ELSE'),
    ('right', 'NOT', 'UMINUS'),
    ('left',  'OR'),
    ('left',  'AND'),
    ('nonassoc', 'EQ', 'NEQ'),
    ('nonassoc', 'LT', 'LEQ', 'GT', 'GEQ'),
    ('left',  'PLUS', 'MINUS'),
    ('left',  '*', '/', '%'),
)
def p_empty(p):
    '''empty :'''
    p[0] = []

def p_program(p):
    '''program : INT MAIN "(" ")" "{" declarations statements "}"'''
    p[0] = Program(p[6], p[7])

def p_declarations(p):
    '''declarations : declaration declarations
                    | empty'''
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = []

def p_declaration(p):
    '''declaration : type ID ";"'''
    p[0] = Declaration(p[1], p[2])

def p_type(p):
    '''type : INT
            | BOOL
            | FLOAT
            | CHAR'''
    p[0] = p.slice[1].type

def p_statements(p):
    '''statements : statement statements
                  | empty'''
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = []


def p_statement(p):
    '''statement : ';'
                 | block
                 | assignment
                 | if_statement
                 | while_statement'''
    p[0] = p[1]

def p_block(p):
    '''block : '{' statements '}' '''
    p[0] = Block(p[2])

def p_assignment(p):
    '''assignment : ID '=' expression ';' '''
    p[0] = Assignment(p[1], p[3])

def p_if_statement(p):
    '''if_statement : IF '(' expression ')' statement ELSE statement
                    | IF '(' expression ')' statement'''
    if len(p) == 8:
        p[0] = IfStatement(p[3], p[5], p[7])
    else:
        p[0] = IfStatement(p[3], p[5], None)

def p_while_statement(p):
    '''
    while_statement : WHILE '(' expression ')' statement
    '''
    p[0] = WhileStatement(p[3], p[5])

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
            p[0] = Literal(p[1], 'INT')
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
  bool flag;
  flag = -flag;
  flag = !flag ;
  y = 2 - 1; 
  x = 0;
  if (x < 3) { x = x + 1; } else { x = x - 1; }
  while (x > 0) { x = x - 1; }
}
"""
lexer  = lex.lex()
parser = yacc.yacc(start='program')
ast = parser.parse(data)

# %%
from llvmlite import ir

intType = ir.IntType(32)
module = ir.Module(name="prog")

fnty = ir.FunctionType(intType, [])
func = ir.Function(module, fnty, name='main')

entry = func.append_basic_block('entry')
builder = ir.IRBuilder(entry)

class IRGenerator(Visitor):
    def __init__(self):
        self.stack = []
        self.symbols = {}
        self.typemap = {
            'INT':   ir.IntType(32),
            'BOOL':  ir.IntType(1),
            'FLOAT': ir.DoubleType(),
            'CHAR':  ir.IntType(8),
        }

    def visit_program(self, node: Program):
        for decl in node.declarations:
            decl.accept(self)

        for stmt in node.statements:
            stmt.accept(self)

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

        then_bb = func.append_basic_block('if.then')
        else_bb = func.append_basic_block('if.else') if node.else_stmt else None
        merge_bb = func.append_basic_block('if.end')

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
        whileHead = func.append_basic_block('while-head')
        whileBody = func.append_basic_block('while-body')
        whileExit = func.append_basic_block('while-exit')
        builder.branch(whileHead)
        builder.position_at_start(whileHead)
        node.condition.accept(self)
        condition = self.stack.pop()
        builder.cbranch(
            condition,
            whileBody,
            whileExit
        )
        builder.position_at_start(whileBody)
        node.statement.accept(self)
        builder.branch(whileHead)
        builder.position_at_start(whileExit)

    def visit_variable(self, node):
        if node.name not in self.symbols:
            raise NameError(f"Undeclared variable '{node.name}'")
        ptr = self.symbols[node.name]
        self.stack.append(builder.load(ptr))
    
    def visit_literal(self, node: Literal):
        self.stack.append(
            ir.Constant(intType, node.value)
        )
    
    def visit_unary_op(self, node: UnaryOp):
        node.operand.accept(self)
        val = self.stack.pop()

        if node.op == '-':
            zero = ir.Constant(val.type, 0)
            self.stack.append(builder.sub(zero, val))
        elif node.op == '!':
            zero = ir.Constant(val.type, 0)
            self.stack.append(builder.icmp_signed('==', val, zero))
        else:
            raise ValueError(f"UnaryOp desconocido {node.op}")

    def visit_binary_op(self, node: BinaryOp) -> None:
        node.lhs.accept(self)
        node.rhs.accept(self)
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        if node.op == '+':
            self.stack.append(builder.add(lhs, rhs))
        elif node.op == '-':
            self.stack.append(builder.sub(lhs, rhs))
        elif node.op == '*':
            self.stack.append(builder.mul(lhs, rhs))
        elif node.op == '/':
            self.stack.append(builder.sdiv(lhs, rhs))
        elif node.op == '%':
            self.stack.append(builder.srem(lhs, rhs))
        elif node.op == '<':
            result = builder.icmp_signed('<', lhs,rhs)
            self.stack.append(result)
        elif node.op == '<=':
            result = builder.icmp_signed('<=', lhs,rhs)
            self.stack.append(result)
        elif node.op == '>':
            result = builder.icmp_signed('>', lhs,rhs)
            self.stack.append(result)
        elif node.op == '>=':
            result = builder.icmp_signed('>=', lhs,rhs)
            self.stack.append(result)
        elif node.op == '==':
            result = builder.icmp_signed('==', lhs,rhs)
            self.stack.append(result)
        elif node.op == '!=':
            result = builder.icmp_signed('!=', lhs,rhs)
            self.stack.append(result)
        elif node.op == '||':
            result = builder.or_(lhs,rhs)
            self.stack.append(result)
        elif node.op == '&&':
            result = builder.and_(lhs,rhs)
            self.stack.append(result)
        else:
            raise ValueError(f"Operador desconocido {node.op}")
        
#%%
ast = parser.parse(data)
visitor = IRGenerator()
ast.accept(visitor)
print(module)
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
