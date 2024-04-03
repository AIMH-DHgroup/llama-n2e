import json

if __name__ == '__main__':

    def check_fails(text):
        global all_fails
        lines = text.split('\n')
        count = 0
        if len(lines) == 0 or len(lines) == 1 or len(lines) == 2:
            for line in lines:
                all_fails.append(line)
            count += 1
        else:
            return 0
        return count

    # insert filename here
    input_filename = "narratives_dict_output-all-lastnewprompt-J98.json"

    with open(input_filename, 'r', encoding='utf-8') as file:
        input_json = json.load(file)

    count_output = {}
    count_fails = {}
    all_fails = []
    for model in input_json.keys():
        count_out = 0
        for narrative in input_json[model]:
            for prompt in input_json[model][narrative]:
                for output in input_json[model][narrative][prompt]:
                    fails = check_fails(input_json[model][narrative][prompt][output])
                    count_out += 1
                    if model not in count_fails:
                        count_fails[model] = fails
                    else:
                        count_fails[model] += fails
        count_output[model] = count_out
    print("Fails: ", count_fails, "\n")
    print("Total saved outputs: ", count_output, "\n")
    print("Printing all outputs...\n")
    for elem in all_fails:
        print(elem)
