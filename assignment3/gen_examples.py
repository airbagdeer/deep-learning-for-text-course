import random

each_sequence_max_length = 10
sequence_max_length = each_sequence_max_length * 9

def generate_custom_sentence(pos_or_neg: str = "pos"):
    def rand_digits():
        return ''.join(random.choices('123456789', k=random.randint(1, each_sequence_max_length)))

    def rand_letters(letter):
        return letter * random.randint(1, each_sequence_max_length)

    if pos_or_neg == "pos":
        sentence = (
                rand_digits() +
                rand_letters('a') +
                rand_digits() +
                rand_letters('b') +
                rand_digits() +
                rand_letters('c') +
                rand_digits() +
                rand_letters('d') +
                rand_digits()
        )
    else:
        sentence = (
                rand_digits() +
                rand_letters('a') +
                rand_digits() +
                rand_letters('c') +
                rand_digits() +
                rand_letters('b') +
                rand_digits() +
                rand_letters('d') +
                rand_digits()
        )

    return sentence

def save_sentences_to_file(filename: str, count: int = 500, pos_or_neg: str = "pos"):
    with open(filename, 'w') as file:
        for _ in range(count):
            file.write(generate_custom_sentence() + '\n')


if __name__ == "__main__":
    save_sentences_to_file("pos_examples", 500, "pos")
    save_sentences_to_file("neg_examples", 500, "neg")