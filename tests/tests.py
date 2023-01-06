import unittest

import re

from pypama import pypama as pm


class TestBasic(unittest.TestCase):

    def test_base(self):
        matched = pm.TokenProvider(['a', 'b', 'c'])
        self.assertEqual(len(matched), 3)
        self.assertEqual(repr(matched), repr(['a', 'b', 'c']))

    def test_sequence(self):
        p = pm.PatternString.make('a')
        m = p.match('a')
        self.assertIsNotNone(m)

    def test_P(self):
        p = pm.build_pattern('<a>')
        m = p.match('a')
        self.assertIsNotNone(m)

    def test_star(self):
        g = pm.PatternStarGreedy.make(
            pm.PatternString('a')
        )

        self.assertIsNotNone(g.match(['a']))

    def test_star2(self):
        g = pm.PatternSeq.make([
            pm.PatternString('a').star(),
            pm.END
        ]
        )
        self.assertIsNone(g.match(['b']))
        self.assertIsNotNone(g.match([]))
        self.assertIsNotNone(g.match(['a', 'a']))

    def test_star3(self):
        gng = pm._build_pattern('<', '(', pm.ANY, '*?', ')', '>')
        g = pm._build_pattern('<', '(', pm.ANY, '*', ')', '>')
        s = list('<body></body>')
        self.assertEqual([list('body></body')], g.match(s).groups())
        self.assertEqual([list('body')], gng.match(s).groups())

    def test_star4(self):
        g = pm.PatternStarGreedy.make(
            pm.PatternString('a')
        )
        self.assertIsNotNone(g.match(['a'] * 30))

    def test_star5(self):
        g = pm._build_pattern('a', '*', 'b', 'c', '*', 'd', 'e', '*')
        self.assertIsNotNone(g.match(list('bcd')))
        self.assertIsNotNone(g.match(list('aaaabccccccccd')))

    def test_star6(self):
        g = pm._build_pattern('c', '*', 'b', '(', 'c', '*', ')', 'd', 'e', '*')
        self.assertIsNone(g.match(list('ccd')))
        self.assertListEqual(g.match(list('ccbccd')).groups(),[['c']*2])

    def test_capture(self):
        g = pm._build_pattern('(', 'a', ')')
        self.assertIsNotNone(g.match(list('a')))
        self.assertEqual(g.match(list('a')).groups(), [['a']])

    def test_capture2(self):
        g = pm._build_pattern('a', '(', '(', 'b', 'c', ')', '*', ')', 'd')
        m = g.match(list('abcbcd'))
        self.assertIsNotNone(m)
        self.assertEqual(m.groups(), [['b', 'c', 'b', 'c'], ['b', 'c']])
        m = g.match(list('ad'))
        self.assertIsNotNone(m)
        self.assertEqual(m.groups(), [[], None])

    def test_capture_backref(self):
        g = pm._build_pattern('(', '.', ')', 'b', '\\1')
        m = g.match(list('aba'))
        self.assertIsNotNone(m)

    def test_detect_cycle(self):
        g = pm.build_pattern(r'(.).*?\1')
        m = g.match(list('abcvc'))
        self.assertIsNone(m)
        m = g.match(list('abcva'))
        self.assertIsNotNone(m)

    def test_detect_cycle2(self):
        g = pm.build_pattern(r'.*(<a>.<b>)(.*?\1)')
        m = g.match(list('rtaebqwaeb'))
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), ['q', 'w', 'a', 'e', 'b'])
        m = g.match(list('a1bcva'))
        self.assertIsNone(m)

    def test_or1(self):
        g = pm._build_pattern('a', '|', 'b', 'd')
        self.assertIsNotNone(g.match(list('ad')))
        self.assertIsNone(g.match(list('ab')))

    def test_or2(self):
        g = pm._build_pattern('a', '|', '(', 'b', 'd', ')', 'c')
        self.assertIsNotNone(g.match(list('ac')))
        self.assertIsNotNone(g.match(list('bdc')))
        self.assertIsNone(g.match(list('bd')))

    def test_opt1(self):
        g = pm._build_pattern('a', '?', 'b')
        self.assertIsNotNone(g.match(list('ab')))
        self.assertIsNotNone(g.match(list('b')))
        self.assertIsNone(g.match(list('aab')))

    def test_mul(self):
        g = pm._build_pattern('a', '{2,4}', 'b')
        self.assertIsNotNone(g.match(list('aab')))
        self.assertIsNotNone(g.match(list('aaaab')))
        self.assertIsNone(g.match(list('ab')))

    def test_mul2(self):
        g = pm._build_pattern('a', '{2}', 'b')
        self.assertIsNotNone(g.match(list('aab')))
        self.assertIsNone(g.match(list('ab')))

    def test_mul3(self):
        g = pm._build_pattern('(', 'a', '{2}', ')', '*', '$')
        self.assertListEqual(g.match(list('aaaa')).groups(), [['a']*2])
        self.assertIsNone(g.match(list('aaa')))

    def test_any(self):
        self.assertIsNotNone(pm._build_pattern(pm.ANY).match(['a']))
        self.assertIsNone(pm._build_pattern(pm.ANY, pm.END).match(['a', 'b']))

    def test_end(self):
        self.assertIsNotNone(pm._build_pattern(pm.END).match([]))
        self.assertIsNotNone(pm._build_pattern(pm.ANY, pm.END).match(['a']))
        self.assertIsNone(pm._build_pattern(pm.END).match(['a', 'b']))

    def test_mix1(self):
        g = pm._build_pattern(
            '(', '.', '*?', 'a', ')', '{2}', 'd'
        )
        self.assertIsNotNone(
            g.match(list('aad'))
        )

    def test_mix2(self):
        g = pm._build_pattern(
            'a', '(', 'b', '|', 'c', 'o', '*', ')', '{2}', 'd'
        )
        self.assertIsNotNone(
            g.match(list('acoobd'))
        )
        self.assertEqual(repr(g), '[a,([b|[c],o*]){[2]},d]')

    def test_end2(self):
        g = pm._build_pattern(
            'a', '?', '{2}', 'b', '$'
        )
        self.assertIsNotNone(g.match(list('b')))
        self.assertIsNotNone(g.match(list('aab')))
        self.assertIsNotNone(g.match(list('ab')))
        self.assertIsNone(g.match(list('bc')))
        self.assertIsNone(g.match(list('aaab')))

    def test_inversible(self):
        g = pm._build_pattern(pm.PatternStarNonGreedy(pm.ANY), ~pm.is_num)
        self.assertIsNotNone(g.match([1, 2, 3, '']))
        self.assertIsNone(g.match([1, 2, 3, 4]))

    def test_example(self):
        example_list = ['a', 'a', 1, '', None, 'b', 'c', 'e']
        g = pm.build_pattern((~pm.is_num).star(False), '(', pm.is_num, ')', '.*', pm.is_none, '(',
                             pm.ANY, pm.ANY, ')')
        self.assertEqual(g.match(example_list).groups(), [[1], ['b', 'c']])
        self.assertIsNone(g.match(['a', 'a', 'b', '', None, 'b', 'c', 'e']))


class TestP(unittest.TestCase):

    def test_split(self):
        self.assertIsNotNone(pm.TOKENS.match('<toto>'))
        self.assertIsNotNone(pm.TOKENS.match('<re:titi>'))
        self.assertIsNotNone(pm.TOKENS.match('*?'))
        self.assertIsNotNone(pm.TOKENS.match('*'))
        self.assertEqual(re.findall(pm.TOKENS, '<toto><re:titi>'), ['<toto>', '<re:titi>'])

    def test_simple_p(self):
        a = pm.build_pattern('<string>')
        self.assertIsInstance(a, pm.PatternSeq)
        self.assertIsInstance(a.patterns[0], pm.PatternString)
        self.assertEqual(a.patterns[0].value, 'string')
        a = pm.build_pattern('<s1><s2>')
        self.assertIsInstance(a, pm.PatternSeq)
        self.assertIsInstance(a.patterns[0], pm.PatternString)
        self.assertEqual(a.patterns[0].value, 's1')
        self.assertEqual(len(a.patterns), 2)
        self.assertEqual(a.patterns[1].value, 's2')

    def test_simple_pattern(self):
        p = pm.build_pattern('<foo>', '<bar>', '<toto>')
        self.assertIsNotNone(p.match(['foo', 'bar', 'toto']))
        self.assertIsNone(p.match(['foo', 'bar', 'titi']))
        p = pm.build_pattern('<c:foo>', foo=lambda x: x < 2)
        self.assertIsNotNone(p.match([1]))
        self.assertIsNone(p.match([4]))

    def test_regex(self):
        a = pm.build_pattern('<re:toto><r:titi>')
        self.assertIsInstance(a.patterns[0], pm.PatternRegex)
        self.assertIsNotNone(a.patterns[0].value.match('toto'))
        self.assertIsInstance(a.patterns[1], pm.PatternRegex)
        self.assertIsNotNone(a.patterns[1].value.match('titi'))

    def test_1(self):
        pat = pm.build_pattern(r"(<re:[\d/]+><re:^(?![\\d%, ]+$).*>*<re:^[\d% ,]+$>*$)")
        self.assertIsNotNone(pat.match(['01/02/2000', 'toto', '10.00']))
        pat = pm.build_pattern(r"(<><re:.+><>*$)")
        self.assertIsNotNone(pat.match(['', 'toto', '']))


class TestFunctional(unittest.TestCase):
    def test_isnum(self):
        self.assertTrue(pm.is_num('0.12'))
        self.assertFalse(pm.is_num('0a.12'))
        self.assertTrue(pm.is_num.__not__()('0a.12'))

    def test_functional(self):
        pat = pm.build_pattern('(', pm.is_num, '*)$')
        self.assertIsNotNone(pat.match(['1.5', '1.2']))

    def test_empty(self):
        pat = pm.build_pattern(r"(<>*$")
        pat2 = pm.PatternSeq.make(
            [
                pm.PatternCapture(
                    pm.PatternSeq(
                        [
                            pm.PatternStarGreedy(
                                pm.PatternString('')
                            ),
                            pm.END
                        ])
                )])
        self.assertEqual(repr(pat), repr(pat2))
        self.assertEqual(pat2.match(['', '']).groups(), [['', '']])

    def test_empty2(self):
        pat = pm.build_pattern(r"(<>*(?P<commentaire>", (lambda x: x != ''), r")<>*$)")
        self.assertEqual(pat.match(['', 'toto']).groupdict(),
                         {'commentaire': ['toto']})


class TestFunctions(unittest.TestCase):
    def test_functions(self):
        @pm.F
        def is_str(x):
            return isinstance(x, str)

        pat = pm.build_pattern('.*(<c:is_str><c:is_int>)', is_str=is_str)
        self.assertEqual(pat.match([2, 2, 4, 'toto', 6, 8]).groups(),
                         [['toto', 6]])
        self.assertIsNone(pat.match([2, 2, 4, 'toto', [], 1]))

    def test_functions1(self):
        @pm.F
        def is_str(x):
            return isinstance(x, str)

        pat = pm.build_pattern('.*(<c:is_str>|<c:is_int>)',
                               lambda x: x == 6, is_str=is_str)
        self.assertEqual(pat.match([2, 2, 4, 'toto', 6, 8]).groups(),
                         [['toto']])
        self.assertIsNone(pat.match([2, 2, 4, 'toto', [], 6, 1]))

    def test_functions2(self):
        @pm.F
        def is_str(x):
            return isinstance(x, str)

        pat = pm.build_pattern('.*(', is_str | pm.is_int, ')',
                               lambda x: x == 6, is_str=is_str)
        self.assertEqual(pat.match([2, 2, 4, 'toto', 6, 8]).groups(),
                         [['toto']])
        self.assertIsNone(pat.match([2, 2, 4, 'toto', [], 6, 1]))

    def test_functions4(self):
        @pm.F
        def is_str(x):
            return isinstance(x, str)

        pat = pm.build_pattern('.*(', is_str & (lambda s: s.startswith('hello')),
                               lambda x: x == 6, ')', is_str=is_str)
        self.assertEqual(pat.match([2, 2, 4, 'hello world', 6, 8]).groups(),
                         [['hello world', 6]])
        self.assertIsNone(pat.match([2, 2, 4, 'toto', [], 6, 1]))


class TestNamedPattern(unittest.TestCase):
    def test_named_capture(self):
        pat = pm.build_pattern('.*(?P<toto><re:bonj.ur>).*')
        a = pat.match(['titi', 'bonjaur'])
        self.assertIsNotNone(a)
        self.assertEqual(a.groupdict(), {'toto': ['bonjaur']})


class TestIncrementalMatch(unittest.TestCase):
    def test_incremental_basic(self):
        pat = pm.build_pattern('a', 'b', 'c')
        a = pm.TokenProvider(['a'])
        self.assertIsNotNone(pat._match(a))
        self.assertIsNone(pat.match(a.fork()))
        a.append_tokens('b', 'c')
        self.assertIsNotNone(pat._match(a))

class TestSplitFind(unittest.TestCase):
    def test_split(self):
        pat = pm.build_pattern(pm.is_int)
        self.assertListEqual(list(pat.split(['toto',1,'titi',2])), [['toto'], ['titi'], []])
        self.assertListEqual(list(pat.split([1,'titi'])), [[], ['titi']])

    def test_split2(self):
        pat = pm.build_pattern("<r:t.t.>(<c:is_int>)")
        self.assertListEqual(
            list(pat.split([
                'toto',1,'part1.1', 'part1.2',
                'titi',2, 'part2.1',
                'tutu',3,
                'toto'])), [[], [1], ['part1.1', 'part1.2'],[2], ['part2.1'], [3], ['toto']])

    def test_find(self):
        pat = pm.build_pattern(pm.is_int)
        self.assertListEqual(list(pat.find(['toto',1,'titi',2])), [[1],[2]])
        
if __name__ == "__main__":
    unittest.makeSuite(TestSplitFind).debug()