import pprint
import random
import rhymer, chainer


with open('trump.txt', 'r') as fo:
	content = fo.read()


r = rhymer.Rhymer(content)
lexicon = r.vocab	
clean_text = r.cleaned_text		# all words in order, lowercase and no punctuation
c = chainer.Markov(None, clean_text)


poem = ""
poem_length = 8 
line_length = 10 

while True:
	for l in range(poem_length):
		while True:
			seed_pos = random.randint(line_length, len(clean_text))
			seed_word = clean_text[seed_pos]
			print("Trying seed word " + seed_word)
			try:
				poss_rhymes = r.rhyme(seed_word)
				next_rhyme = random.choice(poss_rhymes)
				
				if (seed_word == next_rhyme): 
					continue
		
				print("Rhymes with: " + next_rhyme)
				(w1, w2) = seed_word, clean_text[seed_pos-1]
	
				next_rhyme_pos = c.find(clean_text, next_rhyme) 
				(r1, r2) = next_rhyme, clean_text[next_rhyme_pos-1]
	
				line_1 = c.constuct_chain(length=line_length, seed_choice=(w1,w2)) + '\n'
				line_2 = c.constuct_chain(length=line_length, seed_choice=(r1,r2)) + '\n'

				poem += line_1 + line_2
				break
	
			except:
				print("no rhyme for " + seed_word)
				continue
	print('\n')
	print(poem)

	raw_input("Another poem?> ")
	poem = ''
