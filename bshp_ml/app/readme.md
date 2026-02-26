# Документация API python-сервиса

## `/`

### GET

Main Page

Root method returns html ok description
@return: HTML response with ok micro html

#### Responses

| Code | Description |
|------|-------------|
| `200` | Successful Response |


## `/health`

### GET

Health

#### Responses

| Code | Description |
|------|-------------|
| `200` | Successful Response |

#### Example request

```bash
curl -X GET "http://localhost:8000/health" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## `/version`

### GET

Version

#### Responses

| Code | Description |
|------|-------------|
| `200` | Successful Response |

#### Example request

```bash
curl -X GET "http://localhost:8000/version" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## `/save_data`
