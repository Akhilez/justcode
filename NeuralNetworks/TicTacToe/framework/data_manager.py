import json


class DataManager:

    def __init__(self, file_name='data.json', max_size=10):
        self.file_name = file_name
        self.max_size = max_size

    def write(self, matches):
        with open(self.file_name, 'w') as data:
            data.write(json.dumps({'matches': matches}))

    def enqueue(self, matches):
        old_matches = self.get()
        matches.extend(old_matches)
        self.write(matches[:self.max_size])

    def get(self):
        try:
            with open(self.file_name, 'r') as data:
                data_string = data.read()
                if len(data_string) > 0:
                    return json.loads(data_string)['matches']
        except:
            pass
        return []

    def clear(self):
        try:
            with open(self.file_name, 'w') as data:
                data.write('')
        except FileNotFoundError:
            return
