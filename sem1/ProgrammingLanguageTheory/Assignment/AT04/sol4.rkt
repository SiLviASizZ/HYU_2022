#lang racket
(provide (all-defined-out))

(define (check_bst x)
  (if (null? x) #t
      (letrec ([check_left (lambda(i)
                    (if (null? i) #t (>(car x)(car i))))]
           [check_right (lambda(i)
                    (if (null? i) #t (<(car x)(car i))))])
      (and (check_left(cadr x))(check_right(caddr x))(check_bst(cadr x))(check_bst(caddr x)))
        )
    )
  )

(define (apply f x)
  (if (null? x) null
      (list (f (car x))(apply f(cadr x))(apply f(caddr x))))
)


(define (equals bst_1 bst_2)
      (letrec ([match (lambda(x1 x2)
                        (if (null? x2) #f
                            (or (= x1 (car x2))(match x1 (cadr x2))(match x1(caddr x2)))))]
               [search (lambda(x1 x2)
                        (if (null? x1) #t
                            (and (match (car x1) x2)(search(cadr x1) x2)(search(caddr x1) x2))))])
      (and (search bst_1 bst_2)(search bst_2 bst_1))
    )
)
