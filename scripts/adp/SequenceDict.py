
from bisect import bisect
from collections import defaultdict

# Adds method at(atkey) that returns the value at the nearest key <= atkey
# When size exceeds maxlen, entries with the smallest keys are discarded
class SequenceDict(defaultdict):
    def __init__(self, default_factory=None, maxlen=None):
        defaultdict.__init__(self, default_factory)
        self.maxlen = maxlen
    def at(self, s):
        """Access value at the nearest key<=s
            or the s'th highest key if s is a negative integer.
            E.g., {6:'a', 8:'b'}.at(7)  -> 'a' (searches for 7)
                {6:'a', 8:'b'}.at(-1) -> 'b' (indexes with -1)
        """
        seq = sorted(self.keys())
        if s < 0:
            try:
                return self[seq[s]]
            except TypeError:
                pass
        idx = bisect(seq, s) - 1
        if idx == -1:
            raise KeyError
        return self[seq[idx]]
    def __getitem__(self, s):
        """Deletes smallest key if size exceeds self.maxlen"""
        item = defaultdict.__getitem__(self, s)
        if self.maxlen is not None and len(self) > self.maxlen:
            seq = sorted(self.keys())
            del self[seq[0]]
        return item
    def __setitem__(self, s, val):
        """Deletes smallest key if size exceeds self.maxlen"""
        ret = defaultdict.__setitem__(self, s, val)
        if self.maxlen is not None and len(self) > self.maxlen:
            seq = sorted(self.keys())
            del self[seq[0]]
        return ret

if __name__ == "__main__":
    s = Sequence()
    s[0] = 'a'
    s[2] = 'c'
    assert len(s) == 2
    assert s.at(1) == s[0] == 'a'
    s = Sequence(list)
    s[0].append('hello')
    s[0].append('world')
    assert len(s) == 1
    assert s.at(10) == ['hello', 'world']
    s = Sequence(maxlen=2)
    s[100] = 0
    s[101] = 1
    s[102] = 2
    s[103] = 3
    assert len(s) == 2
    assert s.at(102) == 2
    assert s.at(200) == 3
    try:
        s.at(101)
    except KeyError:
        pass
    else:
        assert(False)
    print "All tests PASSED"
    
