import json

if __name__ == '__main__':

    def check_broken_sentences(text):
        global all_sentences
        lines = text.split('\n')
        count_broken_sentences = 0
        for line in lines:
            if line:
                if not line.endswith(('.', '!', '?')):
                    count_broken_sentences += 1
                    all_sentences.append(line)
        return count_broken_sentences

    # insert filename here
    input_filename = "narratives_dict_output-all-lastnewprompt-J98.json"

    with open(input_filename, 'r', encoding='utf-8') as file:
        input_json = json.load(file)

    count_output = {}
    count_broken = {}
    all_sentences = []
    for model in input_json.keys():
        count_out = 0
        for narrative in input_json[model]:
            for prompt in input_json[model][narrative]:
                for output in input_json[model][narrative][prompt]:
                    broken_sentences = check_broken_sentences(input_json[model][narrative][prompt][output])
                    count_out += 1
                    count_broken[model] = broken_sentences
        count_output[model] = count_out
    print("Broken sentences: ", count_broken, "\n")
    print("Total saved outputs: ", count_output, "\n")
    print("Printing all sentences...\n")
    for elem in all_sentences:
        print(elem)
