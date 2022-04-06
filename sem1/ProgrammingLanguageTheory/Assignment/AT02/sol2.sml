(* Problem 1 *)
datatype expr = NUM of int
            | PLUS of expr * expr
            | MINUS of expr * expr;

datatype formula = TRUE
            | FALSE
            | NOT of formula
            | ANDALSO of formula * formula
            | ORELSE of formula * formula
            | IMPLY of formula * formula
            | LESS of expr * expr;
                (* LESS(a, b) is true if a < b *)

(* eval function , takes a formula value and returns Boolean value of formula , formula -> bool *)
(* inplies => false only if ( True, False )  *)

fun eval (x:formula) =
    case x of
        TRUE => true
        | FALSE => false
        | NOT(x1) => if eval x1 then false else true
        | ANDALSO(x1, x2) => if eval x1 then eval x2 else false
        | ORELSE(x1, x2) => if eval x1 then true else eval x2
        | IMPLY(x1, x2) => if eval x1 then eval x2 else true
        | LESS(x1, x2) => 
                let
                fun useExpr (x:expr) =
                    case x of
                        NUM(x1) => x1
                        | PLUS(x1, x2) => useExpr(x1) + useExpr(x2)
                        | MINUS(x1, x2) => useExpr(x1) - useExpr(x2)
                in
                    if useExpr(MINUS(x1, x2)) < 0 then true else false
                end;
                (* How to put expr in here ? => make expr function ? *)
(* in LESS function, Do I have to consider int type value as an input ? => it is weird to operate with only BOOL type input *)

(* Problem 2 *)
(* checkMetro : metro -> bool *)
(* What is expected output ?  *)
(* if not true, print Error Message station "a" is not in area "a" *)

type name = string;
datatype metro = STATION of name
                | AREA of name * metro
                | CONNECT of metro * metro;
(*
fun checkMetro (x:metro) =
    case x of
        STATION x1 => 
        | AREA(x1, x2) =>
        | CONNECT(x1, x2) => 
*)

(* Problem 3 *)
datatype 'a lazyList = nullList 
                    | cons of 'a * (unit -> 'a lazyList); (* lazy evaluation ? => First Wrap with function , call when needed *)
(*
fun useLL (x:'a lazyList) =
    case x of
        nullList => x
        | cons(x1, x2) => cons(x1, x2);
*)
fun seq(first:int, last:int) =
    if first = last then cons(first, fn() => nullList)
    else cons(first, fn() => seq(first+1, last));
    (* Without anonymous function, We cannot Delay Execution ? Got idea of using anonymous function from jeong-min Yu, who graduated HYU in 2021 *)

fun infseq(first:int) =
    cons(first, fn() => infseq(first+1));

fun firstN(lazyListVal, n:int) = (* return list, first n elements . if n>elements => return ALL *)
    case (lazyListVal, n) of
        (x1, 0) => []
        | (nullList, x1) => []
        | (cons(x1, x2), y1) => x1::firstN(x2(), y1-1);

fun Nth(lazyListVal, n:int) = (* return int, at nth place . n>elements => return NONE using option *)
    case (lazyListVal, n) of
        (nullList, x1) => NONE
        | (cons(x1, x2), y1) => if y1 = 1 then SOME x1 else Nth(x2(), y1-1); (* USE SOME => 'a option *)

fun filterMultiples(lazyListVal, n) = (* remove n, n*2, n*3 .... from lazyListVal *)
    case (lazyListVal, n) of
        ()