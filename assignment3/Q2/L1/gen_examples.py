import random
import string

each_sequence_max_length = 20
each_sequence_min_length = 10
sequence_max_length = each_sequence_max_length * 9
amount_of_b_for_positive = int((each_sequence_min_length + each_sequence_max_length) / 2)

def generate_custom_sentence(pos_or_neg: str = "pos"):
    def rand_digits(min_length = each_sequence_min_length, max_length = each_sequence_max_length):
        return ''.join(random.choices('123456789', k=random.randint(min_length, max_length)))

    # def random_letter():
    #     return random.choice(string.ascii_letters)

    def rand_letters(letter, pos_or_neg):
        if pos_or_neg == "pos":
            # return rand_digits(1,3) + letter * amount_of_b_for_positive + rand_digits(1,3)
            return letter * amount_of_b_for_positive
        else:
            return letter * random.randint(each_sequence_min_length, each_sequence_max_length)

    if pos_or_neg == "pos":
        sentence = (
                rand_digits() +
                rand_letters('b', 'pos') +
                rand_digits() +
                rand_letters('b', 'pos') +
                rand_digits() +
                rand_letters('b', 'pos') +
                rand_digits() +
                rand_letters('b', 'pos') +
                rand_digits()
        )
    else:
        flag = True
        while flag:
            sentence = (
                    rand_digits() +
                    rand_letters('b', 'neg') +
                    rand_digits() +
                    rand_letters('b', 'neg') +
                    rand_digits() +
                    rand_letters('b', 'neg') +
                    rand_digits() +
                    rand_letters('b', 'neg') +
                    rand_digits()
            )
            amount_of_b = sentence.count('b')
            if amount_of_b != amount_of_b_for_positive * 4:
                flag = False

    return sentence

def save_sentences_to_file(filename: str, count: int = 500, pos_or_neg: str = "pos"):
    with open(filename, 'w') as file:
        for _ in range(count):
            file.write(generate_custom_sentence() + '\n')


if __name__ == "__main__":
    print(generate_custom_sentence('pos'))
    print(generate_custom_sentence('neg'))
    # save_sentences_to_file("pos_examples", 500, "pos")
    # save_sentences_to_file("neg_examples", 500, "neg")