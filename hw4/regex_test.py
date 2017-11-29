import re

s = 'sstttrreeeeetttccchhhhh'
m = re.sub(r'(([a-z])\2{2,})', r'\g<2>', s)
print(m)
m = re.sub(r'(([a-z])\2{2,})', r'\g<2>\g<2>', s)
print(m)
m = re.sub(r'(([a-z])\2{1,})', r'\g<2>', s)
print(m)


s = 'stretching'
m = re.sub(r'ing$', r'', s)
print(m)

s = 'exciting'
m = re.sub(r'ing$', r'e', s)
print(m)

s = 'getting'
m = re.sub(r'(([a-z])\2{1,2}ing$)', r'\g<2>', s)
print(m)

s = 'stretched'
m = re.sub(r'ed$', r'', s)
print(m)

s = 'excited'
m = re.sub(r'ed$', r'e', s)
print(m)

'''
    for i, s in enumerate(lines):
        if (i+1)%50000==0: print('Convering line', i+1)
        for j, w in enumerate(s):
            w1 = re.sub(r'(([a-z])\2{2,})$', r'\g<2>', w)
            #w1 = re.sub(r'(([a-z])\2{2,})', r'\g<2>', w) # repeated 3+ times, replace by 1
            #w2 = re.sub(r'(([a-z])\2{2,})', r'\g<2>\g<2>', w) # repeated 3+ times, replace by 2
            #w3 = re.sub(r'(([a-z])\2{1,})', r'\g<2>', w) # repeated 2+ times, replace by 1
        lines[i] = s

    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        if i % 10000 == 0: print(i)
        s1 = re.sub(r'ing$', r'', s) # work -> working
        s2 = re.sub(r'ing$', r'e', s) # excite -> exciting
        s3 = re.sub(r'(([a-z])\2{1,2}ing$)', r'\g<2>', s) # get -> getting
    return lines

'''
