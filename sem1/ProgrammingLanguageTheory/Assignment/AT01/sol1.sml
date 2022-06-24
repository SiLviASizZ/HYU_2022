fun merge (xs: int list, ys: int list) =
    if null xs
    then ys
        else if null ys
        then xs
            else if hd(xs) < hd(ys)
            then hd(xs)::merge(tl(xs), ys)
    else hd(ys)::merge(xs, tl(ys));

fun reverse (xs: int list) =
    let
        fun helper (zs: int list, ys: int list) =
            if null zs
            then ys
            else helper(tl(zs), hd(zs)::ys)
    in
        helper (xs, [])
    end;

fun pi(a:int, b:int, f:int->int) =
    if a = b
    then f(a)
    else f(a)*pi(a+1,b,f)
(*How can I know about input data? cannot test about pi function..*)

fun digits(x: int) =
    let
        fun helper(a:int, b:int list) =
            if ( a = 0 )
            then b
            else helper((a div 10), (a mod 10)::b)
    in
        helper(x, [])
    end;

fun digitalRoot(x:int) =
    let
        fun sum(x:int list) =
            if null x
            then 0
            else hd(x) + sum(tl(x))
        fun divideUntilDigit(x:int) =
            if ( x div 10 ) = 0
            then x
            else divideUntilDigit(sum(digits(x)))
    in
        divideUntilDigit(x)
    end;

fun additivePersistence(x:int) =
    let
        fun sum(x:int list) =
            if null x
            then 0
            else hd(x) + sum(tl(x))
        fun countUntilDigit(x:int, y:int) =
            if ( x div 10 ) = 0
            then y
            else countUntilDigit(sum(digits(x)), y+1)
    in
        countUntilDigit(x, 0)
    end;