


signature MathLib =
sig
    val fact : int -> int
    val half: real -> real 
    val half_pi : real 
    val doubler : int -> int 
end



structure MyMathLib  :> MathLib =
struct
    fun fact(x)=
        if x=0
        then 1
        else x * fact (x - 1)

    fun half (y) = y/2.0 

    val half_pi = Math.pi/2.0

(*   fun doubler (y) = y + y *)
end
