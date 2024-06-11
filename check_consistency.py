import json

if __name__ == '__main__':

    def check_broken_sentences(text, mod):
        global all_sentences
        lines = text.split('\n')
        for line in lines:
            if line:
                if not line.endswith(('.', '!', '?')):
                    all_sentences.append(line)
                    models.append(mod)

    # insert filename here
    input_filename = "narratives_dict_output-write-long-narra.json"

    with open(input_filename, 'r', encoding='utf-8') as file:
        input_json = json.load(file)

    count_output = {}
    all_sentences = []
    models = []
    for model in input_json.keys():
        count_out = 0
        for narrative in input_json[model]:
            for prompt in input_json[model][narrative]:
                for output in input_json[model][narrative][prompt]:
                    check_broken_sentences(input_json[model][narrative][prompt][output], model)
                    count_out += 1
        count_output[model] = count_out
    print("Total saved outputs: ", count_output, "\n")
    print("Printing all sentences...\n")
    for model, elem in zip(models, all_sentences):
        print(model, " --> ", elem)
