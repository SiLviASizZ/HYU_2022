
val my_val = 42;

fun sign_my_val() =
    if my_val > 0
    then print "my_val positive!\n"
    else print "my_val not positive!\n"


fun add_or_mult (f, v) =
    if f 7
    then 	         (* function for mult by v *)
    else 	         (* function for add by v *)

val func1 = add_or_mult(fn (x) => (x mod 7) = 0, 2);

