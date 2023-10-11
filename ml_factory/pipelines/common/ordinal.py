from numpy import nan


class Ordinal:
    
    @staticmethod
    def age_group(v):
        if '18' in v:
            return 0
        if '35' in v: 
            return 1
        elif '45' in v:
            return 2
        elif '55' in v:
            return 3
        return 4

    @staticmethod
    def education(v):
        if type(v) == float:
            return nan
        if '<' in v: 
            return 0
        elif '12' in v:
            return 1
        elif 'Some' in v:
            return 2
        return 3

    @staticmethod
    def ordinal_poverty(v):
        if type(v) == float:
            return nan
        if '<' in v: 
            return 1
        elif '>' in v:
            return 2
        return 0
