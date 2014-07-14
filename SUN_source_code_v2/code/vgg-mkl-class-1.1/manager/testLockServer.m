key = randint(1,1,1000) ;

[d,w]= system(sprintf('./tinyLockClient.py morio.cs.ucla.edu 2000 "test-%d"', key)) ;
fprintf('status: %d\n',d);
fprintf('message: %s\n', w) ;

[d,w]= system(sprintf('./tinyLockClient.py morio.cs.ucla.edu 2000 "test-%d"', key)) ;
fprintf('status: %d\n',d);
fprintf('message: %s\n', w) ;
