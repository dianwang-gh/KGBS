# -*- coding: utf-8 -*-

import csv
from nltk import word_tokenize
from nltk.corpus import stopwords

def to_lower(token_words):
    '''
        Convert to lowercase
    '''
    words_lists = [x.lower() for x in token_words]
    return words_lists

PAEs_one = []
PAEs_double_prefer = []  # Store the first word of two-word compounds
PAEs_double = []
PAEs_three_prefer = []   # Store the first word of three-word compounds
PAEs_three = []
two_repeat_one = []      # This list stores cases where the precursor word of a two-word compound exists as a separate compound
three_repeat_one = []  # This list stores cases where the precursor word of a three-word compound exists as a separate compound

mPAEs_one = []
mPAEs_double_prefer = []  # Store the first word of two-word compounds
mPAEs_double = []
mPAEs_three_prefer = []   # Store the first word of three-word compounds
mPAEs_three = []


with open(r'F:\WD\PAEs text mining\dictionary\PAEs_one.txt', 'r', encoding='utf-8') as read_list:
    for line in read_list.readlines():
        line = line.strip()
        PAEs_one.append(line)
with open(r'F:\WD\PAEs text mining\dictionary\PAEs_double_first.txt', 'r', encoding='utf-8') as read1_list:
    for line in read1_list.readlines():
        line = line.strip()
        PAEs_double_prefer.append(line)
with open(r'F:\WD\PAEs text mining\dictionary\PAEs_double.txt', 'r', encoding='utf-8') as read2_list:
    for line in read2_list.readlines():
        line = line.strip()
        PAEs_double.append(line)
with open(r'F:\WD\PAEs text mining\dictionary\PAEs_three_first.txt', 'r', encoding='utf-8') as read3_list:
    for line in read3_list.readlines():
        line = line.strip()
        PAEs_three_prefer.append(line)
with open(r'F:\WD\PAEs text mining\dictionary\PAEs_three.txt', 'r', encoding='utf-8') as read4_list:
    for line in read4_list.readlines():
        line = line.strip()
        PAEs_three.append(line)

'''
with open(r'F:\WD\list_20220319\two_repeat_one.txt', 'r', encoding='utf-8') as read5_list:
    for line in read5_list.readlines():  
        line = line.strip()
        two_repeat_one.append(line)
with open(r'F:\WD\list_20220319\three_repeat_one.txt', 'r', encoding='utf-8') as read6_list:
    for line in read6_list.readlines():  
        line = line.strip()
        three_repeat_one.append(line)
'''
with open(r'F:\WD\PAEs text mining\dictionary\mPAEs_one.txt', 'r', encoding='utf-8') as read7_list:
    for line in read7_list.readlines():
        line = line.strip()
        mPAEs_one.append(line)
with open(r'F:\WD\PAEs text mining\dictionary\mPAEs_double.txt', 'r', encoding='utf-8') as read8_list:
    for line in read8_list.readlines():
        line = line.strip()
        mPAEs_double.append(line)
with open(r'F:\WD\PAEs text mining\dictionary\mPAEs_double_first.txt', 'r', encoding='utf-8') as read9_list:
    for line in read9_list.readlines():
        line = line.strip()
        mPAEs_double_prefer.append(line)
with open(r'F:\WD\PAEs text mining\dictionary\mPAEs_three.txt', 'r', encoding='utf-8') as read10_list:
    for line in read10_list.readlines():
        line = line.strip()
        mPAEs_three.append(line)
with open(r'F:\WD\PAEs text mining\dictionary\mPAEs_double_first.txt', 'r', encoding='utf-8') as read11_list:
    for line in read11_list.readlines():
        line = line.strip()
        mPAEs_three_prefer.append(line)


PAEs_list = set(PAEs_one)     # Remove duplicate words
PAEs_prefer = set(PAEs_double_prefer)
PAEs_double = set(PAEs_double)
PAEs_prefer_three = set(PAEs_three_prefer)
PAEs_three = set(PAEs_three)
#two_repeat_one = set(two_repeat_one)
#three_repeat_one = set(three_repeat_one)

mPAEs_one = set(mPAEs_one)
mPAEs_double = set(mPAEs_double)
mPAEs_double_prefer = set(mPAEs_double_prefer)
mPAEs_three = set(mPAEs_three)
mPAEs_three_prefer = set(mPAEs_three_prefer)

PAEs_list = to_lower(PAEs_list)     # Convert to lowercase
PAEs_prefer = to_lower(PAEs_prefer)
PAEs_double = to_lower(PAEs_double)
PAEs_prefer_three = to_lower(PAEs_prefer_three)
PAEs_three = to_lower(PAEs_three)
two_repeat_one = to_lower(two_repeat_one)
three_repeat_one = to_lower(three_repeat_one)

mPAEs_one = to_lower(mPAEs_one)
mPAEs_double = to_lower(mPAEs_double)
mPAEs_double_prefer = to_lower(mPAEs_double_prefer)
mPAEs_three = to_lower(mPAEs_three)
mPAEs_three_prefer = to_lower(mPAEs_three_prefer)

sr = stopwords.words('english')
def delete_stopwords(token_words):
    '''
        Remove stopwords
    '''
    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words


characters = [' ',',', '.','DBSCAN', ':', ';', '?','(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','...',
              '^', '{', '}', '‡', '†', '§', '\ue0d5', '■', '►']#'-'


def delete_characters(token_words):
    '''
        Remove special characters and numbers
    '''
    words_list = [word for word in token_words if word not in characters]
    return words_list


def double_lookup(word, read_content):   # Check if the phrase formed by the first word and the next word is in the PAEs dictionary
    sents = word_tokenize(read_content)
    token_words = delete_stopwords(sents)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    word_index = token_words.index(word)
    if word_index == len(token_words)-1:
        return None
    else:
        double_word = word + ' ' + token_words[word_index + 1]
        if double_word in PAEs_double: # Check first word
            return double_word
        else:
            return None

def double_mPAEs_lookup(word, read_content):   #Check if the phrase formed by the first word and the next word is in the mPAEs dictionary
    sents = word_tokenize(read_content)
    token_words = delete_stopwords(sents)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    word_index = token_words.index(word)
    if word_index == len(token_words)-1:
        return None
    else:
        double_word = word + ' ' + token_words[word_index + 1]
        if double_word in mPAEs_double: # Check first word
            return double_word
        else:
            return None

def three_lookup(word, read_content):   # Check if the phrase formed by the first word and the next two words is in the dictionary
    sents = word_tokenize(read_content)
    token_words = delete_stopwords(sents)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    word_index = token_words.index(word)
    if word_index == len(token_words)-2 or word_index == len(token_words)-1:
        return None
    else:
        three_word = word + ' ' + token_words[word_index + 1] + ' ' + token_words[word_index + 2]
        if three_word in PAEs_three:
            return three_word
        else:
            return None

def three_mPAEs_lookup(word, read_content):   # Check if the phrase formed by the first word and the next two words is in the dictionary
    sents = word_tokenize(read_content)
    token_words = delete_stopwords(sents)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    word_index = token_words.index(word)
    if word_index == len(token_words)-2 or word_index == len(token_words)-1:
        return None
    else:
        three_word = word + ' ' + token_words[word_index + 1] + ' ' + token_words[word_index + 2]
        if three_word in mPAEs_three:
            return three_word
        else:
            return None

if __name__ == '__main__':
    PAEs_dict = {}
    filename_PAEs_dict = {}
    PAEs = []
    mPAEs_dict = {}
    filename_mPAEs_dict = {}
    mPAEs = []
    mPAEs2PAEs = {}
    from time import time
    start = time()
    # read_text = read_csv(r'E:\test\pdf2csv_test\test.csv')
    # for i in range(len(read_text)):
    # for i in range(1, 1001):
    #     print(i)
    #     read_content = read_text[i][2]+read_text[i][3]+read_text[i][4]+read_text[i][5]+read_text[i][6]
    with open(r'F:\WD\PAEs text mining\txt2csv_all.csv', 'r', encoding='utf-8') as fout:
        csv_reader = csv.reader(fout)
        y = 0
        i = 1
        for rows in csv_reader:
            if y == 0:     # Skip header row
                pass
            else:
                print(rows[0])
                once_PAEs = []
                once_mPAEs = []
                if y % 2000 == 0:
                    filename1 = f'F:\WD\PAEs text mining\PAEs_2{y}.csv'
                    with open(filename1, 'a', newline='') as fout1:
                        csv_writer = csv.writer(fout1)
                        data1 = ['PAEs', 'Count', 'Literature']
                        csv_writer.writerow(data1)
                        for key in PAEs_dict.keys():
                            data1 = [key, PAEs_dict[key]['num'], PAEs_dict[key]['liter']]
                            csv_writer.writerow(data1)
                    print(f'Saved {y} records to F:\WD\PAEs text mining\PAEs_2{y}.csv')
                    filename2 = f'F:\WD\PAEs text mining\mPAEs_2{y}.csv'
                    with open(filename2, 'a', newline='') as fout2:
                        csv_writer = csv.writer(fout2)
                        data2 = ['mPAEs', 'Count', 'Literature']
                        csv_writer.writerow(data2)
                        for key in mPAEs_dict.keys():
                            data2 = [key, mPAEs_dict[key]['num'], mPAEs_dict[key]['liter']]
                            csv_writer.writerow(data2)
                    print(f'Saved {y} records to F:\WD\PAEs text mining\mPAEs_2{y}.csv')
                read_content = rows[2] + rows[3] + rows[4]# Read columns 3, 4, and 5 of each row
                read_content = read_content.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('{', '').replace(
                            '}', '')
                #print(read_content)
                sents = word_tokenize(read_content)
                token_word = delete_stopwords(sents)
                token_word = delete_characters(token_word)
                token_word = to_lower(token_word)
                #print(token_word)


                '''
                f = open(r'E:\Desktop\新建文件夹\delete_test.txt', 'w', encoding='utf-8')
                #for line in newlist:
                f.writelines(newlist)
                f.close()'''
                for word in set(token_word):
                    if word in PAEs_prefer: # Check PAEs first word
                        double_word = double_lookup(word, read_content)
                        if double_word is not None:
                            once_PAEs.append(double_word)
                            if double_word not in PAEs:
                                PAEs.append(double_word)
                                PAEs_dict[double_word] = {'num': 1, 'liter': [rows[1]]}
                            else:
                                PAEs_dict[double_word]['num'] += 1
                                PAEs_dict[double_word]['liter'].append(rows[1])
                        '''else:
                            if word in two_repeat_one:
                                once_chemical.append(word)
                                if word not in PAEs:
                                    PAEs.append(word)
                                    PAEs_dict[word] = {'num': 1, 'liter': [rows[1]]}
                                else:
                                    PAEs_dict[word]['num'] += 1
                                    PAEs_dict[word]['liter'].append(rows[1])'''
                    elif word in PAEs_prefer_three:
                        three_word = three_lookup(word, read_content)
                        if three_word is not None:
                            once_PAEs.append(three_word)
                            if three_word not in PAEs:
                                PAEs.append(three_word)
                                PAEs_dict[three_word] = {'num': 1, 'liter': [rows[1]]}
                            else:
                                PAEs_dict[three_word]['num'] += 1
                                PAEs_dict[three_word]['liter'].append(rows[1])
                        '''else:
                            if word in three_repeat_one:
                                once_PAEs.append(word)
                                if word not in chemicals:
                                    PAEs.append(word)
                                    PAEs_dict[word] = {'num': 1, 'liter': [rows[1]]}
                                else:
                                    PAEs_dict[word]['num'] += 1
                                    PAEs_dict[word]['liter'].append(rows[1])'''
                    elif word in PAEs_list:
                        once_PAEs.append(word)
                        if word not in PAEs:
                            PAEs.append(word)
                            PAEs_dict[word] = {'num': 1, 'liter': [rows[1]]}
                        else:
                            PAEs_dict[word]['num'] += 1
                            PAEs_dict[word]['liter'].append(rows[1])
                    '''----------------------------------Two Library Boundary------------------------------------------'''
                    if word in mPAEs_double_prefer: # mPAEs
                        double_word = double_mPAEs_lookup(word, read_content)
                        if double_word is not None:
                            once_mPAEs.append(double_word)
                            if double_word not in mPAEs:
                                mPAEs.append(double_word)
                                mPAEs_dict[double_word] = {'num': 1, 'liter': [rows[1]]}
                            else:
                                mPAEs_dict[double_word]['num'] += 1
                                mPAEs_dict[double_word]['liter'].append(rows[1])
                        '''else:
                            if word in two_repeat_one:
                                once_mPAEs.append(word)
                                if word not in mPAEs:
                                    mPAEs.append(word)
                                    mPAEs_dict[word] = {'num': 1, 'liter': [rows[1]]}
                                else:
                                    mPAEs_dict[word]['num'] += 1
                                    mPAEs_dict[word]['liter'].append(rows[1])'''
                    elif word in mPAEs_three_prefer:
                        three_word = three_mPAEs_lookup(word, read_content)
                        if three_word is not None:
                            once_mPAEs.append(three_word)
                            if three_word not in mPAEs:
                                mPAEs.append(three_word)
                                mPAEs_dict[three_word] = {'num': 1, 'liter': [rows[1]]}
                            else:
                                mPAEs_dict[three_word]['num'] += 1
                                mPAEs_dict[three_word]['liter'].append(rows[1])
                        '''else:
                            if word in three_repeat_one:
                                once_mPAEs.append(word)
                                if word not in mPAEs:
                                    mPAEs.append(word)
                                    mPAEs_dict[word] = {'num': 1, 'liter': [rows[1]]}
                                else:
                                    mPAEs_dict[word]['num'] += 1
                                    mPAEs_dict[word]['liter'].append(rows[1])'''
                    elif word in mPAEs_one:
                        once_mPAEs.append(word)
                        if word not in mPAEs:
                            mPAEs.append(word)
                            mPAEs_dict[word] = {'num': 1, 'liter': [rows[1]]}
                        else:
                            mPAEs_dict[word]['num'] += 1
                            mPAEs_dict[word]['liter'].append(rows[1])
                    else:
                        pass
                if len(once_PAEs)>0 or len(once_mPAEs)>0:
                    with open(r'E:\Desktop\filename_PAEs_mPAEs.csv',
                              'a', encoding='utf-8', newline='') as writ1:# WD：分两列
                        csv_writer = csv.writer(writ1)
                        data1 = [rows[0], rows[1], once_PAEs, once_mPAEs]
                        csv_writer.writerow(data1)
                    with open(r'E:\Desktop\filename_PAEs-mPAEs.csv',
                            'a', encoding='utf-8', newline='') as writ2:
                        csv_writer = csv.writer(writ2)
                        once_PAEs_tuple = tuple(once_PAEs)
                        #once_mPAEs_tuple = tuple(once_mPAEs)
                        for word_mPAEs in once_mPAEs:
                            mPAEs2PAEs = {once_PAEs_tuple: word_mPAEs}
                            data2 = [rows[0], rows[1], mPAEs2PAEs]
                            csv_writer.writerow(data2)
                    with open(r'E:\Desktop\filename_mPAEs2PAEs.csv',
                            'a', encoding='utf-8', newline='') as writ3:
                        csv_writer = csv.writer(writ3)
                        for word_mPAEs2 in once_mPAEs:
                            for word_PAEs in once_PAEs:
                                data5 = [i, word_mPAEs2, word_PAEs, rows[0]]
                                csv_writer.writerow(data5)
                                i += 1
            y += 1
        stop = time()
        print(stop-start)
        with open(r'E:\Desktop\PAEs_count_liter.csv',
                  'a', encoding='utf-8', newline='') as fout2:
            csv_writer = csv.writer(fout2)
            data3 = ['PAEs', 'Count', 'Literature']
            csv_writer.writerow(data3)
            for key in PAEs_dict.keys():
                data3 = [key, PAEs_dict[key]['num'], PAEs_dict[key]['liter']]
                csv_writer.writerow(data3)
        with open(r'E:\Desktop\mPAEs_count_liter_2.csv',
                  'a', encoding='utf-8', newline='') as fout3:
            csv_writer = csv.writer(fout3)
            data4 = ['mPAEs', 'Count', 'Literature']
            csv_writer.writerow(data4)
            for key in mPAEs_dict.keys():
                data4 = [key, mPAEs_dict[key]['num'], mPAEs_dict[key]['liter']]
                csv_writer.writerow(data4)
        print('done')

