
fun foo(x:'a, y:'a) = [x, y];

val myfoo: int*int->int list = foo;


(* arg type: covariant  *)
fun doubleLength(lenfun:'a list->int, xs) =
  2*lenfun(xs);

fun doubleLength2(lenfun:int list->int, xs) =
  2*lenfun(xs);


doubleLength(length, ["a","b","c"]);
doubleLength2(length, [1, 2, 3]);