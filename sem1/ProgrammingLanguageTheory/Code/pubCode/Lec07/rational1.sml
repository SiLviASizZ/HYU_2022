signature RATIONAL_A = 
sig
  datatype rational = Frac of int * int | Whole of int
  exception BadFrac
  val make_frac : int * int -> rational
  val add : rational * rational -> rational
  val toString : rational -> string
end



structure Rational1 :> RATIONAL_A =
struct
(* Invariant 1: all denominators > 0
   Invariant 2: rationals kept in reduced form *)

  datatype rational = Whole of int | Frac of int*int
  exception BadFrac

(* gcd and reduce help keep fractions reduced, 
   but clients need not know about them *)
(* they _assume_ their inputs are not negative *)
  fun gcd (x,y) =
       if x=y
       then x
       else if x < y
       then gcd(x,y-x)
       else gcd(y,x)

 (* Frac(5, 10) ==> Frac(1, 2)
    Frac(10, 5) ==> Whole(2) *)
   fun reduce r =
       case r of
           Whole _ => r
         | Frac(x,y) => 
           if x=0
           then Whole(0)
           else let val d = gcd(abs x, y) (* using invariant 1 *)
           in 
             if y=d
             then Whole(x div d)
             else Frac(x div d, y div d)
           end

(* when making a frac, we ban zero denominators *)
   fun make_frac (x,y) =
     (* enforce invariant 1 (denom > 0) *)
     (* must handle div by zero *)
     if y=0
     then raise BadFrac
     else if y < 0
       then reduce Frac(~x, ~y)
       else reduce Frac(x, y)


(* using math properties, both invariants hold of the result
   assuming they hold of the arguments *)
   fun add (r1,r2) = 
       case (r1,r2) of
           (Whole(i),Whole(j))   => Whole(i+j)
         | (Whole(i),Frac(j,k))  => (*i+j/k ==> (i*k+j)/k *) Frac(i*k+j, k) 
         | (Frac(j,k),Whole(i))  => Frac(i*k+j, k)
         | (Frac(a,b),Frac(c,d)) => (* a/b+c/d = (a*d+c*b)/bd *)
                                    reduce Frac(a*d+c*b, b*d)
                                  

(* given invariant, prints in reduced form *)
   fun toString r =
       case r of
           Whole i => Int.toString i (* 1=>"1", 2=>"2" *)
         | Frac(a,b) => (Int.toString a) ^ "/" ^ (Int.toString b)
                   (*1/2 ==> "1/2" *)

end
