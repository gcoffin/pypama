======
pypama
======

How to install
--------------

``pip install pypama``

Presentation
------------

This package provides a pattern matching for list of objects. Just as Regex provides regular expression for
strings, this package provides regular expression for other type of lists.

An example is worth a thousand words: assume you have a list

>>> example_list = ['a', 'a', 1, '', None, 'b', 'c',  'e']

For some reason, you know that there is an int and a None, and you
want to extract that number and the 2 strings following the None

>>> from pypama import build_pattern, is_int, is_none
>>> g = build_pattern((~is_int).star(False), '(', is_int, ')', '.*', is_none, '(',ANY,ANY, ')')
>>> g.match(example_list).groups()
[[1], ['b', 'c']]



- ``~is_int`` will matching anything that's not an integer
- ``.star(False)``: equivalent to ``*?`` in regular expressions: repeat as many as necessary
- parenthesis are for capturing groups
- ``.*`` is short for ``ANY.star()`` (match anything, repeatedly)
  
Therefore the pattern above reads as follow: match anything that's not an int, repeatedly, 
until you find an int that you capture in group 1. Then match anything until you
find a None. That must be followed by two elements that you capture in group 2.
  
This is therefore very similar to the ``re`` package, but applied to a list and with
matching that goes beyond strings.
    
Usage
-----

``build_pattern(*args, **functions)``
where args is a list of strings or ``Pattern`` objects, and functions maps function names to executable.

The args can be a string or a list of strings containing:
 - ``<string>``: will match the string inside <>
 - ``S(string)`` outside of quotes will escape the strings. This allows to match parenthesis for instance
 - callable function: X will match if function(X) returns True
 - ``X?``: will be ignored or match X
 - ``X*``: will match if X matches repeatedly (greedy)
 - ``X*?``: will match if X matches repeatedly (non greedy)
 - ``(X)``: will capture the X in a group
 - ``(P<name>X)``: will capture X in a group named "name"
 - ``X|Y``: will match if X or Y matches
 - ``\n``: will match the nth group previously matched
 - ``<r:X>``: will match if the word matches regexp X
 - ``<re:X>``: same as ``<r:X>``
 - ``<c:function>`` or ``<call:function>``: will match if function(X) matches. If function does not exist in the namespace of pypama, add the definition in the function call eg ``build_pattern('<c:foo>', foo=lambda x:x<2)`` will match [1]
 - ``.`` will match any item
 - ``$`` will match if at the end of the list
 - ``X{n}`` will match X exactly n times. n can be a list of integer
 - ``X!``: will match if the item doesn't match X (X must match exactly 1 element)


The following expressions are equally valid and evaluate to the same result;
 - ``build_pattern('<hello>','<world>','.*')``
 - ``build_pattern('<hello><world>.*')``
 - ``S('hello') + S('world') + ANY.star()``
