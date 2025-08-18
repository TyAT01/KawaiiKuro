import ast
import operator as op
from typing import List, Tuple

# NLP
from nltk.tokenize import word_tokenize
from nltk import pos_tag, sent_tokenize
from nltk.corpus import stopwords

# --- Graceful Degradation for NLTK ---
def safe_word_tokenize(text: str) -> List[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()

def safe_pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    try:
        return pos_tag(tokens)
    except LookupError:
        return [(token, 'NN') for token in tokens]

def safe_stopwords() -> set:
    try:
        return set(stopwords.words('english'))
    except LookupError:
        return {'a', 'an', 'the', 'in', 'on', 'of', 'is', 'it', 'i', 'you', 'he', 'she', 'we', 'they', 'my', 'is', 'not'}

def safe_sent_tokenize(text: str) -> List[str]:
    try:
        return sent_tokenize(text)
    except LookupError:
        return text.split('.')

# -----------------------------
# Utils: Safe Math Evaluator
# -----------------------------
ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.FloorDiv: op.floordiv,
}
ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}

class MathEvaluator:
    def eval(self, expr: str) -> str:
        try:
            node = ast.parse(expr, mode='eval')
            value = self._eval(node.body)
            # format cleanly, avoid excessive precision
            if isinstance(value, float):
                if value.is_integer():
                    value = int(value)
                else:
                    value = round(value, 6)
            return f"{expr} = {value}. *smart waifu pose*"
        except Exception:
            return "Math error~ Try easier, my love!"

    def _eval(self, node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
            return ALLOWED_BINOPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARYOPS:
            return ALLOWED_UNARYOPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Expression):
            return self._eval(node.body)
        raise ValueError("Disallowed expression")
