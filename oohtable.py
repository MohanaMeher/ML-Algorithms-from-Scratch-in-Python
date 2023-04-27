class HashTable:
    
    def __init__(self, nbuckets):
        """list of nbuckets lists"""
        self.table = [[] for _ in range(nbuckets)]


    def __len__(self):
        """ 
        Keys in the hashable
        """
        return sum([len(buc) for buc in self.table if len(buc) != 0])
    

    def hashcode(self, o):
        """
        Returns a hashcode for strings and integers; all others returns None
        For integers, returns the integer value.
        For strings, performs operation h = h*31 + ord(c) for all characters in the string
        """
        type_o = type(o)
        if type_o is int:
            return o
        elif type_o is str and len(o) != 0:
            h = 0
            for c in o:
                h = h*31 + ord(c)
            return h
        else:
            return None


    def __setitem__(self, key, value):
        """
        Performs the equivalent of table[key] = value
        Finds the appropriate bucket indicated by key and then appends (key,value)
        to that bucket if the (key,value) pair doesn't exist yet in that bucket.
        If the bucket for key already has a (key,value) pair with that key,
        then replaces the tuple with the new (key,value).
        """
        #h = hash(key) % len(self.buckets)
        bucket_index = self.bucket_indexof(key)
        if bucket_index is None:
            self.table[self.hashcode(key) % len(self.table)].append((key, value))
        else:
            self.table[self.hashcode(key) % len(self.table)][bucket_index] = (key, value)
 

    def __getitem__(self, key):
        """
        Returns the equivalent of table[key].
        Finds the appropriate bucket indicated by the key and looks for the
        association with the key. Returns None if key not found else value
        """
        #h = hash(key) % len(self.buckets)
        bucket_index = self.bucket_indexof(key)
        if bucket_index is None:
            return None
        des_val = self.table[self.hashcode(key) % len(self.table)][bucket_index]
        if des_val is not None:
            return des_val[1]
        else:
            return None



    def __contains__(self, key):
        if self.bucket_indexof(key) is None:
            return False
        return True
        


    def __iter__(self):
        keys = []
        for buc in self.table:
            for item in buc:
                keys.append(item[0])
        return iter(keys)


    def keys(self):
        """
        Returns all keys in the hashtable
        """
        keys = []
        for buc in self.table:
            for item in buc:
                keys.append(item[0])
        return keys


    def items(self):
        """
        Returns all values in the hashable
        """
        values = []
        for buc in self.table:
            for item in buc:
                values.append(item)
        return values


    def __repr__(self):
        """
        Returns a string representing the various buckets of this table.
        The output looks like below:
            0000->
            0001->
            0002->
            0003->parrt:99
            0004->
        where parrt:99 indicates an association of (parrt,99) in bucket 3.
        """
        result = ''
        for i in range(len(self.table)):
            result += '0'*(4-len(str(i))) + str(i) + '->'
            bucket = self.table[i]
            if len(bucket) != 0:
                for j in range(len(bucket) - 1):
                    tup = bucket[j]
                    result += str(tup[0]) + ':' + str(tup[1]) + ', '
                tup = bucket[len(bucket) - 1]
                result += str(tup[0]) + ':' + str(tup[1])
            result+='\n'
        return result



    def __str__(self):
        """
        Returns what str(table) would return for a regular Python dict
        such as {parrt:99}. Order is preserved.
        """
        result = '{'
        for bucket in self.table:
            for j in range(len(bucket)):
                tup = bucket[j]
                result += str(tup[0]) + ':' + str(tup[1]) + ', '
        result = result.strip(', ')
        result += '}'
        return result



    def bucket_indexof(self, key):
        """
        Returns the index of the element within a specific bucket; the bucket is:
        table[hashcode(key) % len(table)]. Linear search is implemented to
        search the bucket to find the tuple containing key.
        """
        bucket = self.table[self.hashcode(key) % len(self.table)]
        for i in range(len(bucket)):
            if bucket[i][0] == key:
                return i
        return None
