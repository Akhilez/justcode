% Try to create dynamic states for blocks world

:- dynamic on/2.
:- dynamic clear/1.
:- dynamic mov/2.

clear(t).

exec_state([]).
exec_state([H|T]):-
  assert(H),
  exec_state(T).

clear_pairs(C1, C2):-
  clear(C1),
  clear(C2),
  not(on(C1, C2)),
  not(C1 = t),
  C1 \== C2.

list_state():-
  listing(on),
  listing(clear).

append_clear_bottom(State, B, State):-
  on(B, t).
append_clear_bottom(State, B, State2):-
  on(B, X),
  not(X = t),
  append(State, [clear(X)], State2).

assert_transition(State, B1, B2):-
  delete(State, on(B1, _), State2),
  delete(State2, clear(B2), State3),
  append(State3, [on(B1,B2)], State4),
  append_clear_bottom(State4, B1, State5),
  write(State),nl,
  write([B1, B2]),nl,
  write(State5), nl,nl,
  assert(mov(State, State5)).

check_equivalence([H|[]], Goal):-
  member(H, Goal).
check_equivalence([H|T], Goal):-
  member(H, Goal),
  check_equivalence(T, Goal).

assert_child_states(State):-
  exec_state(State),
  forall(clear_pairs(C1, C2), assert_transition(State, C1, C2)),
  delete_state(State).

delete_state([Pred|Preds]):-
  retract(Pred),
  delete_state(Preds).
delete_state([]).

%-------------------------BFS------------------------------

go(Start, Goal) :-
	empty_stack(Empty_been_list),
	stack(Start, Empty_been_list, Been_list),
	path(Start, Goal, Been_list).
	% path implements a depth first search in PROLOG
	% Current state = goal, print out been list

path(Goal, Goal, Been_list) :-
  write("The solution is: "),nl,
	reverse_print_stack(Been_list).

path(State, Goal, Been_list) :-
  assert_child_states(State),
	mov(State, Next),
	% not(unsafe(Next)),
	not(member_stack(Next, Been_list)),
	stack(Next, Been_list, New_been_list),
	path(Next, Goal, New_been_list), !.

reverse_print_stack(S) :-
	empty_stack(S).
reverse_print_stack(S) :-
	stack(E, Rest, S),
	reverse_print_stack(Rest),
	write(E), nl.


%%%%%%%%%%%%%%%%%%%% stack operations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % These predicates give a simple, list based implementation of stacks

    % empty stack generates/tests an empty stack

member(X,[X|_]).
member(X,[_|T]):-member(X,T).

empty_stack([]).

    % member_stack tests if an element is a member of a stack

member_stack(E, S) :- member(E, S).

    % stack performs the push, pop and peek operations
    % to push an element onto the stack
        % ?- stack(a, [b,c,d], S).
    %    S = [a,b,c,d]
    % To pop an element from the stack
    % ?- stack(Top, Rest, [a,b,c]).
    %    Top = a, Rest = [b,c]
    % To peek at the top element on the stack
    % ?- stack(Top, _, [a,b,c]).
    %    Top = a

stack(E, S, [E|S]).

%%%%%%%%%%%%%%%%%%%% queue operations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % These predicates give a simple, list based implementation of
    % FIFO queues

    % empty queue generates/tests an empty queue


empty_queue([]).

    % member_queue tests if an element is a member of a queue

member_queue(E, S) :- member(E, S).

    % add_to_queue adds a new element to the back of the queue

add_to_queue(E, [], [E]).
add_to_queue(E, [H|T], [H|Tnew]) :- add_to_queue(E, T, Tnew).

    % remove_from_queue removes the next element from the queue
    % Note that it can also be used to examine that element
    % without removing it

remove_from_queue(E, [E|T], T).

append_queue(First, Second, Concatenation) :-
    append(First, Second, Concatenation).

%%%%%%%%%%%%%%%%%%%% set operations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % These predicates give a simple,
    % list based implementation of sets

    % empty_set tests/generates an empty set.

empty_set([]).

member_set(E, S) :- member(E, S).

    % add_to_set adds a new member to a set, allowing each element
    % to appear only once

add_to_set(X, S, S) :- member(X, S), !.
add_to_set(X, S, [X|S]).

remove_from_set(_, [], []).
remove_from_set(E, [E|T], T) :- !.
remove_from_set(E, [H|T], [H|T_new]) :-
    remove_from_set(E, T, T_new), !.

union([], S, S).
union([H|T], S, S_new) :-
    union(T, S, S2),
    add_to_set(H, S2, S_new).

intersection([], _, []).
intersection([H|T], S, [H|S_new]) :-
    member_set(H, S),
    intersection(T, S, S_new),!.
intersection([_|T], S, S_new) :-
    intersection(T, S, S_new),!.

set_diff([], _, []).
set_diff([H|T], S, T_new) :-
    member_set(H, S),
    set_diff(T, S, T_new),!.
set_diff([H|T], S, [H|T_new]) :-
    set_diff(T, S, T_new), !.

subset([], _).
subset([H|T], S) :-
    member_set(H, S),
    subset(T, S).

equal_set(S1, S2) :-
    subset(S1, S2), subset(S2, S1).

%%%%%%%%%%%%%%%%%%%%%%% priority queue operations %%%%%%%%%%%%%%%%%%%

    % These predicates provide a simple list based implementation
    % of a priority queue.

    % They assume a definition of precedes for the objects being handled

empty_sort_queue([]).

member_sort_queue(E, S) :- member(E, S).

insert_sort_queue(State, [], [State]).
insert_sort_queue(State, [H | T], [State, H | T]) :-
    precedes(State, H).
insert_sort_queue(State, [H|T], [H | T_new]) :-
    insert_sort_queue(State, T, T_new).

remove_sort_queue(First, [First|Rest], Rest).

% --------------------END QUEUE----------------------------

%-------------Queries--------------------

?- go([on(a,t), on(b,a), on(c,b), clear(c)], [on(c, t), clear(a), on(b,c), on(a,b)]).
