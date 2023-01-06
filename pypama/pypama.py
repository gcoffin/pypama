"""
A pattern matching library (for things that are not characters)

>>> from pypama import build_pattern
>>> build_pattern('(<hello>).*(<world>)').match(['hello','bonjour','world']).groups()
[['hello'], ['world']]

"""
import re
from typing import List, Optional


class PatternParseError(Exception):
    def __init__(self, message, token, cursor, pattern):
        self.message = message
        self.token = token
        self.cursor = cursor
        self.pattern = pattern

    def __str__(self):
        return f'''Pattern error {self.message} on token {self.token}
{self.pattern[:self.cursor]} --- {self.pattern[self.cursor:]}'''


class TokenProvider:
    """wraps the list to be matched and provides tokens"""

    def __init__(self, token_list, cursor=0, captures=None):
        self.token_list = token_list
        self.cursor = cursor
        if captures is None:
            self.captures = {}
        else:
            self.captures = captures.copy()

    def get(self):
        result = self.peek()
        self.cursor += 1
        return result

    def peek(self):
        return self.token_list[self.cursor]

    def fork(self):
        return TokenProvider(self.token_list, self.cursor, self.captures.copy())

    def copy(self):
        return TokenProvider(self.token_list[:], self.cursor, self.captures.copy())

    def __len__(self):
        return len(self.token_list) - self.cursor

    def check_not_empty(self):
        return not self.is_empty()

    def is_empty(self):
        return self.cursor == len(self.token_list)

    def __repr__(self):
        return repr(self.token_list[self.cursor:])

    def append_tokens(self, *a):
        self.token_list.extend(a)

    def sync_with(self, o: 'TokenProvider'):
        '''update cursor position and captures based on another TokenProvider'''
        assert self.token_list == o.token_list
        self.cursor = o.cursor
        self.captures.update(o.captures)

    def get_between(self, o):
        assert self.token_list == o.token_list
        assert 0 <= self.cursor <= o.cursor
        return self.token_list[self.cursor:o.cursor]


class MatchObject:
    '''result of a match'''

    def __init__(self, context, captures, token_list):
        self.context = context
        self.captures = captures
        self.token_list = token_list

    def groups(self):
        """Get all captured groups in a list"""
        result = []
        for i in range(len(self.context.groups)):
            result.append(self.captures.get(i, None))
        return result

    def group(self, number: int):
        """Get capture group by index"""
        return self.captures[number - 1]

    def __repr__(self):
        return "<MatchObject>"

    def groupdict(self):
        """Get all captured named groups in a dictionary"""
        return dict((i, j)
                    for i, j in zip(self.context.groups, self.groups())
                    if i is not None and j is not None)

    def __getitem__(self, name):
        """Get captured group by name"""
        return self.captures[self.context.groups.index(name)]


class PatternContext:
    """store capture groups as they appear in the pattern.
    This is done when the pattern is parsed, not during the matching"""

    def __init__(self):
        self.groups = []

    def add_group(self, name=None):
        self.groups.append(name)
        return len(self.groups) - 1


class Pattern:
    """Base class for patterns"""
    follow: 'Pattern'

    @classmethod
    def make(cls, *args):
        result = cls(*args)
        result.set_context(PatternContext())
        return result

    def __init__(self):
        self.follow = None

    def set_follow(self, follow):
        self.follow = follow

    def set_context(self, pattern_context):
        self.context = pattern_context

    def __or__(self, a):
        return PatternOr(self, a)

    def __and__(self, a):
        return PatternAnd(self, a)

    def __add__(self, a):
        return PatternSeq([self, a])

    def star(self, greedy=True):
        """implement the * and *? operator"""
        if greedy:
            return PatternStarGreedy(self)
        else:
            return PatternStarNonGreedy(self)

    def opt(self):
        """implement the ? operator"""
        return PatternOptional(self)

    def capture(self, name:Optional[str]=None):
        """capture with optional name"""
        return PatternCapture(self, name)

    def match(self, arg):
        """check if matches the provided sequence list"""
        if not isinstance(arg, TokenProvider):
            arg = TokenProvider(arg)
        if self._match(arg):
            return MatchObject(self.context, arg.captures, arg.token_list)
        else:
            return None

    def find(self, x:TokenProvider):
        if not isinstance(x, TokenProvider):
            x = TokenProvider(x)
        while not x.is_empty():
            x0 = x.fork()
            m = self._match(x0)
            if m:
                if self.context.groups:
                    yield m.group(1)
                else:
                    yield x.get_between(x0)
                x.sync_with(x0)
            else:
                x.get()

    def split(self, x:TokenProvider):
        if not isinstance(x, TokenProvider):
            x = TokenProvider(x)
        x0 = x.fork()
        while not x.is_empty():
            x1 = x.fork()
            if m:=self.match(x1):
                yield x0.get_between(x)
                if self.context.groups:
                    yield m.group(1)
                x.sync_with(x1)
                x0.sync_with(x1)
            else:
                x.get()
        yield x0.get_between(x)

    def _match(self, x, with_follow=True):
        raise NotImplementedError()

    def _match_follow(self, arg):
        """check if the following token matches the following pattern. 
        Required for pattern sequences containing multiple starred patterns. In
        some instances this implementation is not sufficient."""
        if not self.follow:
            return True  
        arg = arg.fork()
        return self.follow._match(arg) and self.follow._match_follow(arg)


class InversiblePattern:
    """pattern that can implement the "not match" operator"""
    def __invert__(self):
        return PatternInv(self)


class PatternString(Pattern, InversiblePattern):
    """matches a string"""
    def __init__(self, value):
        super().__init__()
        self.value = value

    def _match(self, x, with_follow=True):
        x0 = x.fork()
        if (x0.check_not_empty() and
                self.value == x0.get()):
            x.sync_with(x0)
            return True
        return False

    def __repr__(self):
        return self.value

    def clone(self):
        return self.__class__(self.value)

S = PatternString


class PatternRegex(Pattern, InversiblePattern):
    def __init__(self, value):
        super().__init__()
        if isinstance(value, str):
            value = re.compile(value)
        self.value = value

    def _match(self, x, with_follow=True):
        x0 = x.fork()
        if (x0.check_not_empty()):
            val = x0.get()
            if isinstance(val, str) and self.value.match(val):
                x.sync_with(x0)
                return True
        return False

    def __repr__(self):
        return f'[re:{self.value.pattern}]'

    def clone(self):
        return self.__class__(self.value)


class PatternSeq(Pattern):
    """A sequence of patterns - must match them all
    """
    patterns: List[Pattern]

    def __init__(self, patterns):
        super().__init__()
        self.patterns = list(patterns)
        for i, j in zip(self.patterns[:-1], self.patterns[1:]):
            i.set_follow(j)

    def __repr__(self) -> str:
        return repr(self.patterns)

    def clone(self):
        return self.__class__(i.clone() for i in self.patterns)

    def set_follow(self, follow):
        if not self.patterns:
            raise ValueError('Empty sequence')
        self.patterns[-1].set_follow(follow)

    def _match_follow(self, x):
        return self.patterns[-1]._match_follow(x)

    def set_context(self, pattern_context):
        super().set_context(pattern_context)
        for i in self.patterns:
            i.set_context(pattern_context)

    def _match(self, x: TokenProvider, with_follow=True):
        x0 = x.fork()
        for pattern in self.patterns:
            if not pattern._match(x0, with_follow):
                return False
        x.sync_with(x0)
        return True

    def __repr__(self):
        return f'[{",".join(repr(i) for i in self.patterns)}]'


class PatternOr(Pattern):
    patterns: List[Pattern]

    def __init__(self, *args):
        super().__init__()
        self.patterns = args

    def clone(self):
        return self.__class__(*[i.clone() for i in self.patterns])

    def set_follow(self, follow):
        for i in self.patterns:
            i.set_follow(follow)

    def set_context(self, pattern_context):
        super().set_context(pattern_context)
        for i in self.patterns:
            i.set_context(pattern_context)

    def _match(self, x, with_follow=True):
        for p in self.patterns:
            x0 = x.fork()
            if p._match(x0):
                x.sync_with(x0)
                return True
        return False

    def __repr__(self):
        return '|'.join(repr(i) for i in self.patterns)


class PatternInv(Pattern):
    pattern: Pattern

    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def _match(self, x, with_follow=True):
        x0 = x.fork()
        if not self.pattern._match(x0):
            if not x.check_not_empty():
                return False
            x.get()
            return True
        return False

    def clone(self):
        return self.__class__(self.pattern)


class PatternOptional(Pattern):
    pattern: Pattern

    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def set_context(self, pattern_context):
        super().set_context(pattern_context)
        self.pattern.set_context(pattern_context)

    def _match(self, x, with_follow=True):
        x0 = x.fork()
        if self.pattern._match(x0) and self._match_follow(x0):
            x.sync_with(x0)
            return True
        if self._match_follow(x):
            return True
        return False

    def __repr__(self):
        return f'{self.pattern}?'

    def clone(self):
        return self.__class__(self.pattern)


class PatternAny(Pattern):
    def _match(self, x, with_follow=True):
        try:
            x.get()
            return True
        except IndexError:
            return False

    def __repr__(self):
        return 'ANY'

    def clone(self):
        return self.__class__()


class PatternEnd(Pattern):
    def _match(self, x, with_follow=True):
        return x.is_empty()

    def __repr__(self):
        return "$"

    def clone(self):
        return self.__class__()

class PatternMult(Pattern):
    pattern: Pattern

    def __init__(self, pattern, mults):
        super().__init__()
        self.mults = sorted(mults, reverse=True)
        self.pattern = pattern

    def set_context(self, pattern_context):
        super().set_context(pattern_context)
        self.pattern.set_context(pattern_context)

    def _match(self, x, with_follow=True):
        def m(x0, cnt):
            xf = x0.fork()
            # FOLLOW is either follow or this.pattern
            if cnt <= max(self.mults) and self.pattern._match(xf) and m(xf, cnt + 1):
                x0.sync_with(xf)
                return True
            if cnt in self.mults and self._match_follow(x0):
                return True  # cnt in self.mults
            return False

        return m(x, 0)

    def clone(self):
        return self.__class__(self.pattern.clone(), self.mults)

    def __repr__(self):
        return f'{self.pattern}{{{self.mults}}}'


class PatternStarGreedy(Pattern):
    pattern: Pattern

    def __init__(self, p: Pattern):
        super().__init__()
        self.pattern = p

    def set_context(self, pattern_context):
        super().set_context(pattern_context)
        self.pattern.set_context(pattern_context)

    def _match(self, x, with_follow=True):
        def m(x1):
            x0 = x1.fork()
            if self.pattern._match(x0) and m(x0):
                x1.sync_with(x0)
                return True
            if self._match_follow(x1):
                return True
            return False

        return m(x)

    def __repr__(self):
        return f'{repr(self.pattern)}*'

    def clone(self):
        return self.__class__(self.pattern.clone())


class PatternStarNonGreedy(Pattern):
    pattern: Pattern

    def __init__(self, p):
        super().__init__()
        self.pattern = p

    def set_context(self, pattern_context):
        super().set_context(pattern_context)
        self.pattern.set_context(pattern_context)

    def _match(self, x, with_follow=True):
        def m(x1, level=0):
            if self._match_follow(x1):
                return True
            x2 = x1.fork()
            if self.pattern._match(x2) and m(x2, level + 1):
                x1.sync_with(x2)
                return True
            return False

        return m(x)

    def __repr__(self):
        return f'{repr(self.pattern)}*?'

    def clone(self):
        return self.__class__(self.pattern.clone())


class PatternCapture(Pattern):
    name: Optional[str]
    pattern: Pattern
    group: Optional[int]

    def __init__(self, pattern: Pattern, name: Optional[str] = None):
        self.name = name
        self.pattern = pattern
        self.group = None
        super().__init__()

    def set_follow(self, follow):
        super().set_follow(follow)
        self.pattern.set_follow(follow)

    def set_context(self, pattern_context):
        super().set_context(pattern_context)
        self.group = pattern_context.add_group(self.name)
        self.pattern.set_context(pattern_context)

    def _match(self, x: TokenProvider, with_follow=True):
        start = x.cursor
        result = self.pattern._match(x)
        if result:
            x.captures[self.group] = x.token_list[start:x.cursor]  
        return result

    def __repr__(self):
        return f'({self.pattern})'

    def clone(self):
        return self.__class__(self.pattern.clone(), self.name)


class PatternMatchCapture(Pattern):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.capture = int(re.match(r'\\(\d+)', value).group(1))

    def _match(self, x, with_follow=True):
        x0 = x.fork()
        to_match = x.captures[self.capture - 1]
        for token in to_match:
            if (not x0.check_not_empty() or
                    token != x0.get()):
                return False
        x.sync_with(x0)
        return True

    def __repr__(self):
        return f'\\{self.capture}'

    def clone(self):
        return self.__class__(self.value)

class PatternAnd(Pattern):
    patterns: List[Pattern]

    def __init__(self, *args):
        super().__init__()
        self.patterns = args

    def set_follow(self, follow):
        for i in self.patterns:
            i.set_follow(follow)

    def set_context(self, pattern_context):
        super().set_context(pattern_context)
        for i in self.patterns:
            i.set_context(pattern_context)

    def _match(self, x, with_follow=True):
        x0 = None
        if not self.patterns:
            return True
        for p in self.patterns:
            x0 = x.fork()
            if not p._match(x0):
                return False
        x.sync_with(x0)
        return True

    def __repr__(self):
        return '&'.join(repr(i) for i in self.patterns)

    def clone(self):
        return self.__class__(*[i.clone() for i in self.patterns])

ANY = PatternAny()
END = PatternEnd()


def _build_pattern(*args, cursor=None, only_one=False, dct=None):
    """Build a pattern from a list of tokens
    - *args: list of tokens
    - dct: namespace where to find functions
    - cursor: position of the reader. start with [0] (or None)
    - only_one: will match only one token (for internal purpose)
    """
    dct = dct or {}
    res = []
    if not cursor:
        cursor = [0]
    t = None
    try:
        while True:
            if cursor[0] >= len(args):
                break
            t = args[cursor[0]]
            if not isinstance(t, str):
                if isinstance(t, Pattern):
                    res.append(t.clone())
                elif callable(t):
                    res.append(F(t))
                else:
                    raise ValueError(t)
            else:
                if t == '*':
                    res[-1] = res[-1].star(greedy=True)
                elif t == '*?':
                    res[-1] = res[-1].star(greedy=False)
                elif re.match(r'{[\d,]+}', t):
                    g = re.match(r'{([\d,]+)}', t).group(1)
                    mults = [int(i.strip()) for i in g.split(',')]
                    res[-1] = PatternMult(res[-1], mults)
                elif t == '?':
                    res[-1] = res[-1].opt()
                elif t == '.':
                    res.append(ANY)
                elif t == '$':
                    res.append(END)
                elif t == '!':
                    res[-1] = ~res[-1]
                elif re.match(r'\((?:\?P<(\w+)>)?$', t):
                    name, = re.match(r'\((?:\?P<(\w+)>)?$', t).groups()
                    cursor[0] += 1
                    res.append(PatternCapture(_build_pattern(*args, cursor=cursor, dct=dct), name))
                elif t == ')':
                    break
                elif t == '|':
                    cursor[0] += 1
                    res[-1] = res[-1] | _build_pattern(*args, cursor=cursor, only_one=True, dct=dct)
                elif t == '&':
                    cursor[0] += 1
                    res[-1] = PatternAnd(res[-1],
                                         _build_pattern(*args, cursor=cursor, only_one=True,
                                                        dct=dct))
                elif re.match(r'\\\d+', t):
                    res.append(PatternMatchCapture(t))
                elif t[0] == '<' and t[-1] == '>':
                    t = t[1:-1]
                    m = re.match('([^:]+):(.*)', t)
                    if m:
                        typ, v = m.groups()
                        if typ in ('c', 'call'):
                            res.append(F(eval(v, globals(), dct)))
                        elif typ in ('r', 're'):
                            res.append(PatternRegex(v))
                        else:
                            raise ValueError(r'Unknown {m.groups()}')
                    else:
                        res.append(PatternString(t))
                else:
                    res.append(PatternString(t))
            if only_one:
                break
            cursor[0] += 1
    except PatternParseError:
        raise
    except Exception as e:
        raise PatternParseError(str(e), t, cursor[0], args) from e
    if not res:
        raise PatternParseError('Empty sequence', '', cursor[0], args)
    return PatternSeq.make(res)


TOKENS = re.compile(r"""(?x)
     (
     (?:\*\?)
     |<
        (?:re?:
          |c(?:all)?:
        )?
        (?:[^>]|\\>)*
      >
     |\((?:\?P<\w+>)?
     |\\\d+
     |[$.)!?*|]
     |\w+)""")


def build_pattern(*args, **dct):
    """Use this function to create a pattern"""
    tokens = []
    for arg in args:
        if isinstance(arg, str):
            tokens.extend(TOKENS.findall(arg))
        else:
            tokens.append(arg)
    return _build_pattern(*tokens, dct=dct)


class F(Pattern, InversiblePattern):
    """Turns a function into a pattern (with _match function)"""

    def __init__(self, fun, name=None):
        self.fun = fun
        self.__name__ = name or self.fun.__name__
        super().__init__()

    def clone(self):
        return self.__class__(self.fun, self.__name__)

    def _match(self, x, with_follow=True):
        x0 = x.fork()
        if (x0.check_not_empty() and
                self(x0.get())):
            x.sync_with(x0)
            return True
        return False

    def __call__(self, x):
        return self.fun(x)

    def __or__(self, fun):
        if isinstance(fun, F) or callable(fun):
            return F((lambda x: self.fun(x) or fun(x)), f'{self.__name__}|{fun.__name__}')
        else:
            return super().__or__(fun)

    def __and__(self, fun):
        if isinstance(fun, F) or callable(fun):
            return F((lambda x: self.fun(x) and fun(x)), f'{self.__name__}&{fun.__name__}')
        else:
            return super().__and__(fun)

    def __not__(self):
        return F((lambda x: not self.fun(x)), f'{self.__name__}!')

    def __repr__(self):
        return f'[c:{repr(self.__name__)}]'

    __str__ = __repr__


@F
def is_num(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


@F
def is_int(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


@F
def is_none(s):
    return s is None
