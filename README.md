# Syntactic-semantic-ir

This project parses a subset of C with PLY, builds an AST, generates LLVM IR with llvmlite, JIT-compiles and executes it.

## Repository Structure

```
.
├── analisis.py      # Main parser, AST walker & IR generator
├── arbol.py         # AST node definitions (Visitor pattern)
├── runtime.py       # llvmlite execution engine setup
├── test.c           # Sample C source for demonstration
└── README.md        # This file
```

## Prerequisites

* **Python 3.8+**
* **pip** for installing packages
* **LLVM** (optional, for native `llvmlite` support)

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository**

   ```bash
    git clone https://github.com/RogerHdzC/syntactic-semantic-ir
    cd syntactic-semantic-ir
   ```

2. **Prepare your C source**
   - The default entry is `test.c`. You can modify it or point `analisis.py` at another file.

3. **Run the pipeline**
   ```bash
    python analisis.py
    ````

This will:

* Parse the C source into an AST
* Generate LLVM IR and print it
* JIT-compile and execute the `main` function
* Print both the IR and the program output

## Language Features Supported

* **Types**: `int`, `bool`, `float` (32‑bit), `char`
* **Float literals**: e.g. `3.14f`, `2.0`, `.5`
* **Variables & Declarations**
* **Operators**: `+ - * / %`, comparison, logical (`&& || !`)
* **Assignments**: `=`, `+=`, `-=`
* **Increment/Decrement**: `++`, `--` (prefix & postfix)
* **Control Flow**: `if`/`else`, `while`, `do`/`while`, `for`, `switch`/`case`/`default`, `break`, `return`
* **Function definitions & calls** (including mutual recursion)
* **`printf`** calls supported as an external varargs function

## Extending the Grammar

* **Add tokens** in `analisis.py` (regex, precedence)
* **AST nodes** in `arbol.py`
* **IR emission** in `IRGenerator` methods (`visit_*`)

