#Q 1. Looping:   Printing   all   elements   in   a   certain   range,   Introduction   of   for   operator.   Example-
#  Printing   all   numbers   from   1-10   using   for   loop
#
def Q_1():
    for i in range(1-10):
        print(i)




#Q 2. Finding   if   a   number   is   even   or   odd
def Q_2():
    num = int(input("enter a number(it will be considered as an int) \n :"))
    if num%2 == 0:
        print("Its an Even number")
    else:
        print("Its an Odd number")



#Q 3. Taking   a   range   as   input   from   user   example   user   inputs   a   certain   start   and   end   point   print
# each   element   and   if   it   is   even   or   odd
def Q_3():
    start = int(input("start : "))
    end = int(input("end : "))
    for i in range(start,end+1):
        if i%2 == 0:
            print("{} - {}".format(i,"even"))
        else :
            print("{} - {}".format(i,"odd"))



#Q 4. Modification   to   the   above   task   -   Example-   Print   all   odds   between   2-30
def Q_4():
    start = int(input("start : "))
    end = int(input("end : "))
    print("all odd numbers:\n")
    for i in range(start, end + 1):
        if i % 2 != 0:
            print("{}".format(i))



#Q 5. Round   off   a   number   to   certain   degree   of   precision   [round,   ceil,   floor]
from math import *
def Q_5():
    num = float(input("enter a decimal number : "))
    print("round({}) = {}".format(num,round(num)))
    print("ceil({}) = {}".format(num,ceil(num)))
    print("round({}) = {}".format(num,floor(num)))



#Q 6. Compute   compound   interest   using   loop   for   a   certain   principle   and   interest   amount
def Q_6():
    p = float(input("enter principle : "))
    n = int(input("period : "))
    r = float(input("rate of interest : "))
    for j in range(1,n+1):
        i = p*r/100
        p += i
        print("interest at the end of yr{} = {}".format(j,i))
        print("principle amt at the end of yr{} = {}".format(j,p))



#Q 7. Introduction   of   while:   Example   generate   a   random   value   using   random   module   and   try
# predicting   it   using   guesses   using   while   loop
import random
def Q_7():
    num = random.randint(1,100)
    print("if assigned number is greater that your guess ,'go higher' will be printed ... and like wise")

    while True:
        guess = int(input("Guess : "))
        if num > guess :
            print("go higher")
        elif num < guess:
            print("go lower")
        else:
            print("You got the number : ",num)
            break



#Q 8 Draw a pine tree:
def Q_8():
    h = int(input("height : "))
    for i in range(1,h+1):
        for j in range(1,2*h):
            if j<h+i and j>h-i:
                print("#",end="")
            else :
                print(" ",end="")
        print()
    for i in range(2):
        for j in range(h-1):
            print(" ",end="")
        print("#")



#Q 9. Usage   of   mathematical   functions   in   python   like   math.ceil,   floor,   fabs,   fmod,   trunc,   pow,
# sqrt,   e,   pi   ,log,   natural   log,   degree,   radian
def Q_9():
    print("ceil(2.34) =",ceil(2.34))
    print("floor(2.34) =",floor(2.34))
    print("ceil(2.67) =",ceil(2.67))
    print("floor(2.67) =",floor(2.67))
    print("fabs(2.67) =",fabs(2.67))
    print("fmod(2.67,2) =",fmod(2.67,2))
    print("trunc(2.6776) =",trunc(2.6776))
    print("pow(2.667,2) =",pow(2.667,2))
    print("sqrt(2.667) =",sqrt(2.667))
    print("e =",e)
    print("log10(2.667) =",log10(2.667))
    print("log(2.667) =",log(2.667))
    print("degrees(pi) =",degrees(pi))
    print("radian(180) =",radians(180))



#Q 10.Operations   on   string   using   unicodes   ,splitting   of   string,accessing   elements   of   string   using locations
def Q_10():
    string = input("string : ")
    for i in string:
        print(i,ord(i))




#Q 11. Take   input   from   user   and   translate   it   to   Unicode   and   the   back   to   string
def Q_11():
    name = input("name : ")
    print("Your name in ascii")
    for i in name:
        if i==" ":
            print(" ",end=" ")
        else:
            print(ord(i),end=" ")



#Q 12. Extension   of   same   for   sentences   (I   have   to   go   to   the   doctor)
def Q_12():
    name = input("sentence : ")
    print("Your name in ascii")
    for i in name:
        if i==" ":
            print(" ",end=" ")
        else:
            print(ord(i),end=" ")



#Q 13. Functions   on   string   like   lstrip,rstrip,capitalize,upper,lower
def Q_13():
    string = input()
    print("lstrip : ",string.lstrip())
    print("rstrip : ",string.rstrip())
    print("capitalize : ",string.capitalize())
    print("upper : ",string.upper())
    print("lower : ",string.lower())



#Q 14.  Splitting   and   joining   strings,accessing   each   item   of   a   string   For   example   -   I   have   to   go   to
# a   doctor,   Output   -   I,have,to,   go,   to,   a,   doctor
def Q_14():
    string = input()
    print(",".join(string.split()))



#Q 15 Counting   occurrence   of   a   certain   element   in   a   string,   getting   indexes   that   have   matching
# elements   For   ex   -   Rabbit   count   how   many   times   b   has   occurred
def Q_15():
    string = input("sample : ").lower()
    element = input("element : ").lower()
    count = 0

    for i in range(len(string)-len(element)+1):
        if string[i:i+len(element)] == element:
            count+=1
    print("element repeated",count,"times")



#Q 16 Replacing   one   substring   by   another   For   example   -   Rabbit   -   Replace   ‘bb’   by   ‘cc’
def Q_16():
    string = input("sentence : ")
    old = input("what to replace(case sensitive) : ")
    new = input("by what(case sensitive) : ")

    string = string.replace(old,new)
    print(string)



#Q 17 Acronym   generator   for   any   user   input   (ex-input   is   Random   memory   access   then   output
# should   be   RMA)
def Q_17():
    inp = input("input : ").strip().split()
    for i in inp:
        print(i[0].upper(),end="")



#Q 18  Count   the   number   of   occurrences   of   each   word   in   the   below   given   paragraph   (invariant
# of   upper   case   and   lower   case   characters.)
from collections import *
def Q_18():
    sentence = "The Delhi High Court has restricted the tax authorities from issuing show cause notices to   13 " \
               "banks   for   not   levying   service   tax   on   the   commitment   to   maintain   'minimum   " \
               "average balance'   (MAB)   in   bank   accounts. While   the   government   has   sought   that   show " \
               "  cause notices   to   the   banks   should   prevail,   the   banks   contended   that   if   show " \
               "  cause   notices   were allowed,   they   will   have   to   provision   nearly   Rs   50,000   crore" \
               "   as   tax   and   penalty   in   their books   as   per   accounting   practices.   This   will" \
               "   'wreak   havoc."
    punctuations = ["'",'"',",",".","(",")"]
    sentence = list(sentence)
    for i in sentence:
        if i in punctuations:
            sentence.remove(i)
    sentence = "".join(sentence)
    sentence = sentence.strip().lower().split()

    C = Counter(sentence)
    for i in C:
        print(i,C[i])

