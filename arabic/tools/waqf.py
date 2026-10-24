import re
def waqf(clause):
    c=re.sub(r'[؟?.!]','',clause).strip()
    ws=c.split(); w=ws[-1]
    if re.search('ً[اى]?$',w):
        w=re.sub('ً([اى]?)$',lambda m:'َ'+(m.group(1) or 'ا'),w)
    else:
        m=re.search('[ً-ْ]+$',w); sh=''
        if m:
            sh='ّ' if 'ّ' in m.group(0) else ''
            w=w[:m.start()]
        if w.endswith('ة'): w=w[:-1]+'ه'
        if sh: w=w+sh
        elif not w.endswith(('ا','ى')): w=w+'ْ'
    ws[-1]=w; return ' '.join(ws)
if __name__=='__main__':
    import json
    for s,t in json.load(open('script.json')):
        for p in [p.strip() for p in re.split(r'[،:]',t) if p.strip()]: print(waqf(p).split()[-1])
