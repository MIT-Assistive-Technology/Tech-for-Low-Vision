module Main exposing (..)

import Browser
import Browser.Events
import Element exposing (..)
import Element.Background as Background
import Element.Border as Border
import Element.Events
import Element.Font as Font
import Element.Input as Input
import Html exposing (Html)
import Http
import Json.Decode as Decode
import Url


type alias Flags =
    ( Int, Int )


type Mode
    = Input
    | Display
    | Shortcuts


type alias Model =
    { mode : Mode
    , src : String
    , w : Int
    , h : Int
    , typing : Bool
    , fileInput : String
    , params : Params
    }


type alias Params =
    { poseX : Float
    , poseY : Float
    , poseZ : Float
    , dirX : Float
    , dirY : Float
    , dirZ : Float
    , n : Int
    , i : Int
    }


type Msg
    = None
    | ToDisplay
    | SetTyping Bool
    | FileInput String
    | KeyPressed String
    | Received (Result Http.Error String)


defaults : Flags -> ( Model, Cmd Msg )
defaults flags =
    let
        ( width, height ) =
            flags
    in
    ( { mode = Input
      , src = "nothing"
      , w = width
      , h = height
      , fileInput = ""
      , typing = False
      , params = params
      }
    , Cmd.none
    )


params : Params
params =
    { poseX = 0
    , poseY = 0
    , poseZ = 0
    , dirX = 0
    , dirY = 0
    , dirZ = 1
    , n = 10
    , i = 0
    }


main : Program Flags Model Msg
main =
    Browser.element { init = defaults, update = update, subscriptions = subscriptions, view = view }


getImage : String -> Params -> Cmd Msg
getImage path query =
    let
        baseUrl : String
        baseUrl =
            "http://localhost:5000/get"

        queryParams : String
        queryParams =
            [ ( "file", path )
            , ( "poseX", String.fromFloat query.poseX )
            , ( "poseY", String.fromFloat query.poseY )
            , ( "poseZ", String.fromFloat query.poseZ )
            , ( "dirX", String.fromFloat query.dirX )
            , ( "dirY", String.fromFloat query.dirY )
            , ( "dirZ", String.fromFloat query.dirZ )
            , ( "n", String.fromInt query.n )
            , ( "i", String.fromInt query.i )
            ]
                |> List.map (\( k, v ) -> k ++ "=" ++ Url.percentEncode v)
                |> String.join "&"

        fullUrl =
            baseUrl ++ "?" ++ queryParams
    in
    Http.get { url = fullUrl, expect = Http.expectJson Received imgDecoder }


subscriptions : Model -> Sub Msg
subscriptions _ =
    Browser.Events.onKeyDown (Decode.map KeyPressed keyDecoder)


imgDecoder : Decode.Decoder String
imgDecoder =
    Decode.field "path" Decode.string


keyDecoder : Decode.Decoder String
keyDecoder =
    Decode.field "key" Decode.string


return : Model -> ( Model, Cmd Msg )
return x =
    ( x, Cmd.none )


delta : Float
delta =
    0.05


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    let
        void : ( Model, Cmd Msg )
        void =
            return model

        next : ( Model, Cmd Msg )
        next =
            ( { model | mode = Display, typing = False }, getImage model.fileInput model.params )

        lift : (Params -> Params) -> ( Model, Cmd Msg )
        lift f =
            let
                p : Params
                p =
                    model.params

                new : Model
                new =
                    { model | params = f p }
            in
            ( new, getImage new.fileInput new.params )

        nextFrame : Params -> Params
        nextFrame p =
            { p
                | i =
                    if p.i < p.n then
                        p.i + 1

                    else
                        p.i
            }

        prevFrame : Params -> Params
        prevFrame p =
            { p
                | i =
                    if p.i > 0 then
                        p.i - 1

                    else
                        p.i
            }

        moreFrames : Params -> Params
        moreFrames p =
            { p | n = p.n + 1 }

        lessFrames : Params -> Params
        lessFrames p =
            { p
                | n =
                    if (p.n > p.i) && (p.n > 1) then
                        p.n - 1

                    else
                        p.n
            }

        xUp : Params -> Params
        xUp p =
            { p | poseX = p.poseX + delta }

        xDown : Params -> Params
        xDown p =
            { p | poseX = p.poseX - delta }

        yUp : Params -> Params
        yUp p =
            { p | poseY = p.poseY + delta }

        yDown : Params -> Params
        yDown p =
            { p | poseY = p.poseY - delta }

        zUp : Params -> Params
        zUp p =
            { p | poseZ = p.poseZ + delta }

        zDown : Params -> Params
        zDown p =
            { p | poseZ = p.poseZ - delta }

        dirX : Params -> Params
        dirX p =
            { p | dirX = 1, dirY = 0, dirZ = 0 }

        dirY : Params -> Params
        dirY p =
            { p | dirX = 0, dirY = 1, dirZ = 0 }

        dirZ : Params -> Params
        dirZ p =
            { p | dirX = 0, dirY = 0, dirZ = 1 }
    in
    case msg of
        None ->
            void

        ToDisplay ->
            next

        SetTyping state ->
            return { model | typing = state }

        FileInput input ->
            return { model | fileInput = input }

        KeyPressed key ->
            if model.typing then
                case key of
                    "Enter" ->
                        next

                    _ ->
                        void

            else
                case model.mode of
                    Display ->
                        case key of
                            "n" ->
                                next

                            "b" ->
                                return { model | mode = Input }

                            "?" ->
                                return { model | mode = Shortcuts }

                            "j" ->
                                lift nextFrame

                            "k" ->
                                lift prevFrame

                            "h" ->
                                lift lessFrames

                            "l" ->
                                lift moreFrames

                            "a" ->
                                lift xDown

                            "d" ->
                                lift xUp

                            "w" ->
                                lift yUp

                            "s" ->
                                lift yDown

                            "q" ->
                                lift zUp

                            "e" ->
                                lift zDown

                            "x" ->
                                lift dirX

                            "y" ->
                                lift dirY

                            "z" ->
                                lift dirZ

                            _ ->
                                void

                    _ ->
                        case key of
                            "n" ->
                                next

                            "b" ->
                                return { model | mode = Input }

                            "?" ->
                                return { model | mode = Shortcuts }

                            _ ->
                                void

        Received response ->
            case response of
                Err _ ->
                    void

                Ok path ->
                    return { model | src = path }


view : Model -> Html Msg
view model =
    case model.mode of
        Input ->
            Element.layout [] (inputBox model)

        Display ->
            Element.layout [] (displayBox model)

        Shortcuts ->
            Element.layout [] shortcutBox


vw : Model -> Float -> Float
vw model percent =
    Basics.toFloat model.w * percent / 100


vh : Model -> Float -> Float
vh model percent =
    Basics.toFloat model.h * percent / 100


vw2pt : Model -> Float -> Int
vw2pt model ratio =
    (round << vw model) ratio


vw2px : Model -> Float -> Length
vw2px model ratio =
    px (vw2pt model ratio)


vh2pt : Model -> Float -> Int
vh2pt model ratio =
    (round << vh model) ratio


vh2px : Model -> Float -> Length
vh2px model ratio =
    px (vh2pt model ratio)


white : Color
white =
    rgb255 255 255 255


black : Color
black =
    rgb255 0 0 0


primary : Color
primary =
    rgb255 25 128 230


typed : List (Attribute Msg)
typed =
    [ Element.Events.onFocus (SetTyping True)
    , Element.Events.onLoseFocus (SetTyping False)
    ]


inputBox : Model -> Element Msg
inputBox model =
    column
        [ centerX
        , centerY
        ]
        [ row [ spacing 20 ]
            [ Input.text
                (spacing 20 :: typed)
                { text = model.fileInput
                , placeholder = Nothing
                , onChange = FileInput
                , label = Input.labelAbove [] (text "Enter file name with extension, like model.glb")
                }
            , Input.button
                [ width (px 100)
                , height (px 80)
                , padding 10
                , Background.color primary
                , Font.color white
                , Font.center
                , Border.rounded 20
                ]
                { onPress = Just ToDisplay, label = text "Next" }
            ]
        ]


shortcutBox : Element Msg
shortcutBox =
    column
        [ centerX
        , centerY
        , spacing 10
        ]
        [ text "n: go to model viewer"
        , text "b: change model file"
        , text "j: next frame"
        , text "k: previous frame"
        , text "h: reduce frame count by one"
        , text "l: increase frame count by one"
        , text "d: slightly increase x coordinate"
        , text "a: slightly decrease x coordinate"
        , text "w: slightly increase y coordinate"
        , text "s: slightly decrease y coordinate"
        , text "q: slightly increase z coordinate"
        , text "e: slightly decrease z coordinate"
        , text "x: view cross sections along x axis"
        , text "y: view cross sections along y axis"
        , text "z: view cross sections along z axis"
        , text "?: show this help screen"
        ]


type Order
    = First
    | Second
    | Third


displayBox : Model -> Element Msg
displayBox model =
    let
        w : Float -> Length
        w =
            vw2px model

        h : Float -> Length
        h =
            vh2px model

        p : Params
        p =
            model.params

        floating : Float -> String
        floating num =
            String.fromFloat (toFloat (round (100 * num)) / 100)

        direction : Model -> String
        direction m =
            let
                mp : Params
                mp =
                    m.params

                dx : Float
                dx =
                    abs mp.dirX

                dy : Float
                dy =
                    abs mp.dirY

                dz : Float
                dz =
                    abs mp.dirZ

                imax : Float -> Float -> Float -> Order
                imax a b c =
                    if a > b then
                        if a > c then
                            First

                        else
                            Third

                    else if b > c then
                        Second

                    else
                        Third

                dmax : Order
                dmax =
                    imax dx dy dz
            in
            case dmax of
                First ->
                    "X"

                Second ->
                    "Y"

                Third ->
                    "Z"
    in
    column
        [ width (w 100)
        , height (h 110)
        , Background.color black
        , Font.color white
        ]
        [ image
            [ centerX
            , centerY
            , width (h 90 |> maximum (vw2pt model 90))
            ]
            { src = "../" ++ model.src
            , description = "cross section"
            }
        , el
            [ centerX
            , padding 10
            ]
            (text ("Frame " ++ String.fromInt p.i ++ " of " ++ String.fromInt p.n ++ ", " ++ direction model ++ " direction"))
        , el
            [ centerX
            , paddingEach
                { top = 0
                , bottom = 40
                , left = 0
                , right = 0
                }
            ]
            (text ("Pose (" ++ floating p.poseX ++ ", " ++ floating p.poseY ++ ", " ++ floating p.poseZ ++ ")"))
        ]
