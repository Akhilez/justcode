const http = require('http');

const server = http.createServer((request, response) => {
    response.writeHead(200, {'Context-Type': 'text/plain'});
    response.end('Hello World');
});

server.listen(8004);
