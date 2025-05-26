import random

sequence_min_length = 100
each_sequence_max_length = 150
sequence_max_length = each_sequence_max_length

def generate_custom_sentence(pos_or_neg: str = "pos"):
    def rand_digits(pos_or_neg: str = "pos"):
        return ''.join(random.choices('123456789', k=random.randint(sequence_min_length, each_sequence_max_length if pos_or_neg=='neg' else each_sequence_max_length - 1)))

    def rand_letters(letter):
        return letter * random.randint(1, each_sequence_max_length)

    if pos_or_neg == "pos":
        sentence = (
                'b'+
                rand_digits('pos')
        )
    else:
        sentence = (
                rand_digits('neg')
        )

    return sentence

def save_sentences_to_file(filename: str, count: int = 500, pos_or_neg: str = "pos"):
    with open(filename, 'w') as file:
        for _ in range(count):
            file.write(generate_custom_sentence() + '\n')


if __name__ == "__main__":
    # save_sentences_to_file("pos_examples", 500, "pos")
    # save_sentences_to_file("neg_examples", 500, "neg")
    print(generate_custom_sentence('pos'))
    print(generate_custom_sentence('neg'))