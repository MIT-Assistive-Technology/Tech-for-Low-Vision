module Main exposing (..)

import Browser
import Html exposing (Html, div, text)


type alias Flags =
    ()


type alias Model =
    {}


type Msg
    = None


main : Program Flags Model Msg
main =
    Browser.sandbox { init = {}, update = update, view = view }


update : Msg -> Model -> Model
update msg model =
    case msg of
        None ->
            model


view : Model -> Html Msg
view model =
    div []
        [ text "Placeholder text"
        ]
